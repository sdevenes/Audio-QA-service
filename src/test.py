import whisper_timestamped as whisperT
import json
from pathlib import Path
import textwrap

from langchain.chains import RetrievalQA
from langchain import hub
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline


from langchain.text_splitter import RecursiveCharacterTextSplitter
# for OS != Windows yous should use and adapt the code below
from langchain_community.document_loaders import JSONLoader
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain_huggingface import HuggingFacePipeline

from typing import List, Optional, Union

# for Windows systems you should use the code below
# from tqdm.auto import tqdm


class JSONLoaderWindows(BaseLoader):
    def __init__(self, file_path: Union[str, Path], content_key: Optional[str] = None):
        self.file_path = Path(file_path).resolve()
        self._content_key = content_key

    def create_documents(self, processed_data):
        documents = []
        for item in processed_data:
            content = ''.join(item)
            document = Document(page_content=content, metadata={})
            documents.append(document)
        return documents

    def process_item(self, item, prefix=""):
        if isinstance(item, dict):
            result = []
            for key, value in item.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                result.extend(self.process_item(value, new_prefix))
            return result
        elif isinstance(item, list):
            result = []
            for value in item:
                result.extend(self.process_item(value, prefix))
            return result
        else:
            return [f"{prefix}: {item}"]

    def process_json(self, data):
        if isinstance(data, list):
            processed_data = []
            for item in data:
                processed_data.extend(self.process_item(item))
            return processed_data
        elif isinstance(data, dict):
            return self.process_item(data)
        else:
            return []

    def load(self) -> List[Document]:
        """Load and return documents from the JSON file."""

        docs = []
        with open(self.file_path, 'r') as json_file:
            try:
                data = json.load(json_file)
                processed_json = self.process_json(data)
                docs = self.create_documents(processed_json)
            except json.JSONDecodeError:
                print("Error: Invalid JSON format in the file.")
        return docs


def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text


def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.page_content)
        print(source.metadata.get("source"))
        print(source.metadata.get("start_time"))


def process(audio_filepath, question):

    local_path = "../models/Phi-3-mini-4k-instruct"
    local_path_tokenizer = "../models/Phi-3-mini-4k-instruct_tokenizer"
    # MODEL_NAME = "mistralai/Mistral-7B-v0.1" #"mistralai/Mistral-7B-Instruct-v0.1" # "meta-llama/Llama-2-7b-chat"
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) # "mistralai/Mistral-7B-v0.1"
    # model = AutoModelForCausalLM.from_pretrained(model_id)
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model.save_pretrained(local_path)
    # tokenizer.save_pretrained(local_path_tokenizer)

    print("Loading embedding model..")
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("Loading phi3 model..")
    model = AutoModelForCausalLM.from_pretrained(local_path, local_files_only=True)
    print("Loading tokenizer..")
    tokenizer = AutoTokenizer.from_pretrained(local_path_tokenizer, local_files_only=True)
    print(" -> Done")

    # To download a whisper model locally:
    # from whisper import _download, _MODELS
    # _download(_MODELS["base"], ".", False)

    print("Load Whisper Model..")
    audio = whisperT.load_audio(audio_filepath)
    # model = whisperT.load_model("base")
    whisper_model = whisperT.load_model("../models/whisper_base.pt")
    print("Transcribe audio..")
    result = whisperT.transcribe(whisper_model, audio, language="en")
    print(" -> Done")

    # Save temporary the result in a json file
    jsonpath = "temp.json"

    with open(jsonpath, 'w') as fp:
        json.dump(result, fp)

    # create the length function
    def tiktoken_len(text):
        tokens = tokenizer.tokenize(text)
        return len(tokens)

    # splitting the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,  # number of tokens overlap between chunks
        length_function=tiktoken_len,
        separators=[r'\n\n', r'\n', r'(?<=\. )', r'(?<=\, )', r' ', r'']
    )

    print("Load JSON file..")
    loader = JSONLoader(file_path=jsonpath, jq_schema='.segments[]', text_content=False)
    # loader = JSONLoaderWindows(file_path=jsonpath)
    docs = loader.load()
    # we will store the documents in a list
    documents = []
    chunks = []
    for doc in docs:
        content = json.loads(doc.page_content)
        chunks = text_splitter.split_text(content["text"])
        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata={"source":  audio_filepath,
                                                                    "start_time": content["start"],
                                                                    "end_time": content["end"]}))
    print("Creating Chroma db...")
    vectordb = Chroma.from_documents(documents=documents,
                                     embedding=embedding_model,
                                     persist_directory=None)
    print(" -> Done")

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
    HFpipe = HuggingFacePipeline(pipeline=pipe)

    # from langchain_community.chat_models import ChatOllama
    # llm = ChatOllama(model="mistral") # system=..

    retriever_from_llm = vectordb.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'score_threshold': 0.8})

    rag_prompt = hub.pull("rlm/rag-prompt")

    qa_chain = RetrievalQA.from_chain_type(
        HFpipe,
        retriever=retriever_from_llm,
        chain_type_kwargs={"prompt": rag_prompt},
        return_source_documents=True
    )

    llm_response = qa_chain({"query": question})
    return llm_response


# question = "How to make carbonara?"
question = "Should I add cream ?"

answer = process(audio_filepath=Path("temp.mp3"), question=question)
process_llm_response(answer)
