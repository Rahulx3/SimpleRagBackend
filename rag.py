import hashlib
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

COLLECTION_NAME = "pdf_rag"

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.8-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
REBUILD_DATABASE = os.getenv("REBUILD_DATABASE", "false").lower() in ("true", "1", "yes")


def get_chroma_path() -> Path:
    """
    Returns a guaranteed writable path for ChromaDB SQLite storage.
    Falls back to /tmp/chroma_db if running in a container with restricted directory permissions.
    """
    env_path = os.getenv("CHROMA_PATH")
    if env_path:
        p = Path(env_path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    default_path = BASE_DIR / "chroma_db"
    try:
        default_path.mkdir(parents=True, exist_ok=True)
        test_file = default_path / ".perm_check"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()
        return default_path
    except Exception:
        fallback = Path("/tmp/chroma_db")
        fallback.mkdir(parents=True, exist_ok=True)
        logger.info("Using writable container fallback directory for ChromaDB: %s", fallback)
        return fallback


CHROMA_PATH = get_chroma_path()


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


def get_source_file() -> Path:
    """
    Returns the primary knowledge source. Prefers structured markdown (sample.md)
    over PDF for significantly higher RAG retrieval accuracy.
    """
    md_path = BASE_DIR / "data" / "sample.md"
    if md_path.exists():
        return md_path
    return find_pdf_path()


def get_source_hash() -> str:
    """
    Calculates MD5 hash of the active source file to detect content updates.
    """
    source = get_source_file()
    if source.exists():
        return hashlib.md5(source.read_bytes()).hexdigest()
    return ""


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
    Loads and splits PDF using PyPDFLoader.
    """
    pdf_path = find_pdf_path()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at expected path: {pdf_path}")

    logger.info("Loading PDF from %s", pdf_path)
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    documents = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
    if not documents:
        raise ValueError(f"No readable content extracted from {pdf_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    logger.info("Split %s PDF pages into %s chunks", len(documents), len(chunks))
    return chunks


def load_and_split_documents():
    """
    Loads and splits the knowledge source into semantically coherent chunks.
    If sample.md is present, splits along employee/section boundaries so that
    entire employee monthly histories stay intact in single chunks.
    """
    source = get_source_file()
    if source.suffix.lower() == ".md":
        logger.info("Loading structured markdown knowledge from %s", source)
        content = source.read_text(encoding="utf-8")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=3200,
            chunk_overlap=100,
            separators=["\n---\n\n## ", "\n---\n", "\n## ", "\n\n", "\n"]
        )
        chunks = splitter.create_documents([content], metadatas=[{"source": str(source)}])
        logger.info("Created %s structured chunks from %s", len(chunks), source.name)
        return chunks

    return load_and_split_pdf()


def initialize_vector_db(embeddings: GoogleGenerativeAIEmbeddings) -> Chroma:
    """
    Initializes or loads the Chroma vector database.
    Performs file cleanups BEFORE initializing Chroma to prevent SQLite code 1032 (readonly database).
    """
    source_hash = get_source_hash()
    hash_file = CHROMA_PATH / ".source_hash"
    stored_hash = hash_file.read_text(encoding="utf-8").strip() if hash_file.exists() else ""

    needs_rebuild = REBUILD_DATABASE or (not stored_hash) or (stored_hash != source_hash)

    # Clean existing database files BEFORE opening SQLite connection
    if needs_rebuild and CHROMA_PATH.exists():
        logger.info("Resetting vector database directory before fresh ingestion: %s", CHROMA_PATH)
        shutil.rmtree(CHROMA_PATH, ignore_errors=True)
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)

    # Open Chroma cleanly
    vector_db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PATH)
    )

    # If already populated and hash matches, reuse existing vectors
    if not needs_rebuild and vector_db._collection.count() > 0:
        logger.info(
            "Loaded existing vector database with %s documents.",
            vector_db._collection.count()
        )
        return vector_db

    logger.info("Ingesting structured document chunks into ChromaDB using %s...", EMBEDDING_MODEL)
    chunks = load_and_split_documents()
    batch_size = 2  # Micro-batches to respect Google Free Tier limits

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        for attempt in range(6):
            try:
                vector_db.add_documents(batch)
                time.sleep(1.5)  # Cooldown between batches
                break
            except Exception as e:
                err_msg = str(e)
                if attempt == 5:
                    logger.exception("Failed to add batch starting at index %s after 6 attempts", i)
                    raise
                backoff = 25 if ("429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg) else 5 * (attempt + 1)
                logger.warning(
                    "Rate limit or error adding batch (%s). Waiting %ss for quota recovery (attempt %s/6)...",
                    err_msg[:120],
                    backoff,
                    attempt + 1
                )
                time.sleep(backoff)

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    if source_hash:
        hash_file.write_text(source_hash, encoding="utf-8")

    logger.info("Vector database ingestion complete with %s total items.", vector_db._collection.count())
    return vector_db
