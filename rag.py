import logging
import os
from pathlib import Path
import shutil
import time

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

PDF_PATH = BASE_DIR / "data" / "sample.pdf"
CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "pdf_rag"

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
REBUILD_DATABASE = os.getenv("REBUILD_DATABASE", "false").lower() in ("true", "1", "yes")

_COMPLETE_MARKER = CHROMA_PATH / ".ingestion_complete"


def initialize_models() -> tuple[GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI]:
    """
    Initializes Gemini embeddings and LLM.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY missing from environment variables.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key
    )

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=api_key,
        max_retries=2,
        temperature=0.2
    )

    return embeddings, llm


def load_and_split_pdf():
    """
    Loads and splits the PDF into chunks for vector indexing.
    """
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found at expected path: {PDF_PATH}")

    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    # Filter out empty or whitespace-only pages
    documents = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
    if not documents:
        raise ValueError(f"No readable content extracted from {PDF_PATH}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    logger.info("Split %s pages into %s chunks", len(documents), len(chunks))
    return chunks


def initialize_vector_db(embeddings: GoogleGenerativeAIEmbeddings) -> Chroma:
    """
    Initializes or loads the Chroma vector database. Rebuilds if requested,
    if corrupted, or if there is a dimension mismatch with the active embedding model.
    """
    if REBUILD_DATABASE and CHROMA_PATH.exists():
        logger.info("REBUILD_DATABASE is active. Removing existing vector database at %s", CHROMA_PATH)
        shutil.rmtree(CHROMA_PATH)

    vector_db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PATH)
    )

    # Check if vector DB is already populated and verify embedding compatibility
    if vector_db._collection.count() > 0:
        try:
            # Probe a search to verify dimension compatibility
            vector_db.similarity_search("health check", k=1)
            if not _COMPLETE_MARKER.exists():
                _COMPLETE_MARKER.write_text("ok", encoding="utf-8")
            logger.info(
                "Loaded existing vector database with %s documents.",
                vector_db._collection.count()
            )
            return vector_db
        except Exception as e:
            logger.warning(
                "Existing vector DB is incompatible with the current embedding model (%s). Rebuilding database...",
                e
            )
            if CHROMA_PATH.exists():
                shutil.rmtree(CHROMA_PATH)
            vector_db = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=str(CHROMA_PATH)
            )

    logger.info("Ingesting document chunks into ChromaDB using %s...", EMBEDDING_MODEL)
    chunks = load_and_split_pdf()
    batch_size = 10

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        for attempt in range(5):
            try:
                vector_db.add_documents(batch)
                break
            except Exception:
                if attempt == 4:
                    logger.exception("Failed to add batch starting at index %s after 5 attempts", i)
                    raise
                backoff = 5 * (attempt + 1)
                logger.warning("Error adding batch. Retrying in %ss (attempt %s/5)...", backoff, attempt + 1)
                time.sleep(backoff)

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    _COMPLETE_MARKER.write_text("ok", encoding="utf-8")
    logger.info("Vector database ingestion complete with %s total items.", vector_db._collection.count())

    return vector_db
