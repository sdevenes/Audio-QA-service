import asyncio
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from common_code.config import get_settings
from common_code.http_client import HttpClient
from common_code.logger.logger import get_logger, Logger
from common_code.service.controller import router as service_router
from common_code.service.service import ServiceService
from common_code.storage.service import StorageService
from common_code.tasks.controller import router as tasks_router
from common_code.tasks.service import TasksService
from common_code.tasks.models import TaskData
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from contextlib import asynccontextmanager
import os

# Imports required by the service's model
import json
import whisper_timestamped as whisperT
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from langchain import hub

settings = get_settings()
# classifier = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")


class MyService(Service):
    """
    QA on an audio file using whisper_timestamped for the transcription and huggingface model for QA.
    """

    # Any additional fields must be excluded for Pydantic to work
    _model: object
    _tokenizer: object
    _embedding_model: object
    _logger: Logger

    def __init__(self):
        super().__init__(
            name="Audio-QA",
            slug="audio-qa",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            # TODO: 4. CHANGE THE INPUT AND OUTPUT FIELDS, THE TAGS AND THE HAS_AI VARIABLE
            data_in_fields=[
                FieldDescription(name="text", type=[FieldDescriptionType.TEXT_PLAIN]),
                FieldDescription(name="audio_file", type=[FieldDescriptionType.AUDIO_MP3]),
            ],
            data_out_fields=[
                FieldDescription(name="result", type=[FieldDescriptionType.TEXT_PLAIN]),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.NATURAL_LANGUAGE_PROCESSING,
                    acronym=ExecutionUnitTagAcronym.NATURAL_LANGUAGE_PROCESSING,
                ),
            ],
            has_ai=True,
            docs_url="https://docs.swiss-ai-center.ch/reference/core-concepts/service/",
        )
        self._logger = get_logger(settings)

        local_path = "../models/Phi-3-mini-4k-instruct"
        local_path_tokenizer = "../models/Phi-3-mini-4k-instruct_tokenizer"

        emb_name = "BAAI/bge-base-en"
        # emb_name = "BAAI/bge-large-en-v1.5"
        print("Loading embedding model..")
        self._embedding_model = HuggingFaceBgeEmbeddings(
            model_name=emb_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("Loading phi3 model..")
        self._model = AutoModelForCausalLM.from_pretrained(local_path, local_files_only=True)
        print("Loading tokenizer..")
        self._tokenizer = AutoTokenizer.from_pretrained(local_path_tokenizer, local_files_only=True)
        print(" -> Done")

    # TODO: 5. CHANGE THE PROCESS METHOD (CORE OF THE SERVICE)
    def process(self, data):
        # Get the audio file
        audio = data["audio_file"].data

        # Save the audio file locally
        audio_filepath = "temp.mp3"
        if os.path.exists(audio_filepath):
            os.remove(audio_filepath)

        # raw content as mp3 temp file
        with open(audio_filepath, "wb") as f:
            f.write(audio)

        # Get the text to analyze from storage
        text = data["text"].data
        # Convert bytes to string
        text = text.decode("utf-8")
        # Limit the text to 142 words
        text = " ".join(text.split()[:500])
        print("text: ", text)

        # Run the model
        # To download a whisper model locally:
        # from whisper import _download, _MODELS
        # _download(_MODELS["base"], ".", False)

        print("Load Audio file..")
        audio = whisperT.load_audio(audio_filepath)
        print("Load Whisper Model..")
        whisper_model = whisperT.load_model("../models/whisper_base.pt")
        print("Transcribe audio..")
        result = whisperT.transcribe(whisper_model, audio, language="en")
        print(" -> Done")

        # Save temporary the result in a json file
        jsonpath = "tmp.json"

        with open(jsonpath, 'w') as fp:
            json.dump(result, fp)

        print("Define RecursiveCharacterTextSplitter..")

        # create the length function
        def tiktoken_len(text):
            tokens = self._tokenizer.tokenize(text)
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
                                         embedding=self._embedding_model,
                                         persist_directory=None)
        print(" -> Done")

        pipe = pipeline("text-generation", model=self._model, tokenizer=self._tokenizer, max_new_tokens=100)
        HFpipe = HuggingFacePipeline(pipeline=pipe)

        print("Creating retriever..")
        retriever_from_llm = vectordb.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'score_threshold': 0.8})

        rag_prompt = hub.pull("rlm/rag-prompt")

        qa_chain = RetrievalQA.from_chain_type(
            HFpipe,
            retriever=retriever_from_llm,
            chain_type_kwargs={"prompt": rag_prompt},
            return_source_documents=True
        )
        print("Asking model..")
        llm_response = qa_chain({"query": text})
        print("llm_response: ", llm_response)

        # Convert the result to bytes
        if "result" in llm_response:
            file_bytes = llm_response["result"].encode("utf-8")
        else:
            file_bytes = str(llm_response).encode("utf-8")

        return {
            "result": TaskData(
                data=file_bytes,
                type=FieldDescriptionType.TEXT_PLAIN
            )
        }


service_service: ServiceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Manual instances because startup events doesn't support Dependency Injection
    # https://github.com/tiangolo/fastapi/issues/2057
    # https://github.com/tiangolo/fastapi/issues/425

    # Global variable
    global service_service

    # Startup
    logger = get_logger(settings)
    http_client = HttpClient()
    storage_service = StorageService(logger)
    my_service = MyService()
    tasks_service = TasksService(logger, settings, http_client, storage_service)
    service_service = ServiceService(logger, settings, http_client, tasks_service)

    tasks_service.set_service(my_service)

    # Start the tasks service
    tasks_service.start()

    async def announce():
        retries = settings.engine_announce_retries
        for engine_url in settings.engine_urls:
            announced = False
            while not announced and retries > 0:
                announced = await service_service.announce_service(my_service, engine_url)
                retries -= 1
                if not announced:
                    time.sleep(settings.engine_announce_retry_delay)
                    if retries == 0:
                        logger.warning(
                            f"Aborting service announcement after "
                            f"{settings.engine_announce_retries} retries"
                        )

    # Announce the service to its engine
    asyncio.ensure_future(announce())

    yield

    # Shutdown
    for engine_url in settings.engine_urls:
        await service_service.graceful_shutdown(my_service, engine_url)

# TODO: 6. CHANGE THE API DESCRIPTION AND SUMMARY
api_summary = """
Question answering on an audio file.
"""

api_description = """
Transcribe an audio file with whisper-timestamped, store it into a chromadb vector store and do
question-answering with a model from huggingface (Phi-3-mini-4k-instruct).

This service has two input files:
 - A text file containing the question to ask the model about the audio content.
 - An audio file.

 """

# Define the FastAPI application with information
# TODO: 7. CHANGE THE API TITLE, VERSION, CONTACT AND LICENSE
app = FastAPI(
    lifespan=lifespan,
    title="Audio file Question-Answering API.",
    description=api_description,
    version="0.0.1",
    contact={
        "name": "Swiss AI Center",
        "url": "https://swiss-ai-center.ch/",
        "email": "info@swiss-ai-center.ch",
    },
    swagger_ui_parameters={
        "tagsSorter": "alpha",
        "operationsSorter": "method",
    },
    license_info={
        "name": "GNU Affero General Public License v3.0 (GNU AGPLv3)",
        "url": "https://choosealicense.com/licenses/agpl-3.0/",
    },
)

# Include routers from other files
app.include_router(service_router, tags=["Service"])
app.include_router(tasks_router, tags=["Tasks"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Redirect to docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs", status_code=301)
