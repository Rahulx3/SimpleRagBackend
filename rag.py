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

CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "pdf_rag"

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
REBUILD_DATABASE = os.getenv("REBUILD_DATABASE", "false").lower() in ("true", "1", "yes")


def find_pdf_path() -> Path:
    """
    Locates the PDF file case-insensitively to prevent Linux/Render FileNotFoundError.
    """
    candidates = [
        BASE_DIR / "data" / "Sample.pdf",
        BASE_DIR / "data" / "sample.pdf",
    ]
    for p in candidates:
        if p.exists():
            return p

    data_dir = BASE_DIR / "data"
    if data_dir.exists():
        pdf_files = list(data_dir.glob("*.pdf"))
        if pdf_files:
            return pdf_files[0]

    return candidates[0]


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
    pdf_path = find_pdf_path()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at expected path: {pdf_path}")

    logger.info("Loading PDF from %s", pdf_path)
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    # Filter out empty or whitespace-only pages
    documents = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
    if not documents:
        raise ValueError(f"No readable content extracted from {pdf_path}")

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
    Initializes or loads the Chroma vector database.
    If already populated, reuses the existing vectors without rebuilding.
    """
    if REBUILD_DATABASE and CHROMA_PATH.exists():
        logger.info("REBUILD_DATABASE is active. Removing existing vector database at %s", CHROMA_PATH)
        shutil.rmtree(CHROMA_PATH)

    vector_db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PATH)
    )

    # If the database already has documents, load immediately
    if vector_db._collection.count() > 0:
        logger.info(
            "Loaded existing vector database with %s documents.",
            vector_db._collection.count()
        )
        return vector_db

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

    logger.info("Vector database ingestion complete with %s total items.", vector_db._collection.count())
    return vector_db
