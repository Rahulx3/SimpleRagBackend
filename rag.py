
import os
import time
import shutil
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_chroma import Chroma


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = BASE_DIR / "data" / "sample.pdf"
CHROMA_PATH = str(BASE_DIR / "chroma_db")

COLLECTION_NAME = "pdf_rag"

LLM_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-001"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

NORMAL_TOP_K = 1
BROAD_TOP_K = 20

REBUILD_DATABASE = False


# ==========================================
# INITIALIZE MODELS
# ==========================================

def initialize_models():

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY missing in .env")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        task_type="RETRIEVAL_DOCUMENT",
        google_api_key=api_key
    )

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=api_key,
        max_retries=2
    )

    return embeddings, llm


# ==========================================
# LOAD AND SPLIT PDF
# ==========================================

def load_and_split_pdf():

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    documents = [
        doc for doc in documents
        if doc.page_content.strip()
    ]

    if not documents:
        raise ValueError("No readable PDF content found.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    return splitter.split_documents(documents)


# ==========================================
# INITIALIZE CHROMA
# ==========================================

def initialize_vector_db(embeddings):

    if REBUILD_DATABASE and Path(CHROMA_PATH).exists():
        shutil.rmtree(CHROMA_PATH)

    vector_db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

    if vector_db._collection.count() > 0:
        return vector_db

    chunks = load_and_split_pdf()

    for i in range(0, len(chunks), 10):

        batch = chunks[i:i + 10]

        for attempt in range(5):

            try:
                vector_db.add_documents(batch)
                break

            except Exception:

                if attempt == 4:
                    raise

                time.sleep(15 * (attempt + 1))

    return vector_db


# ==========================================
# QUESTION CLASSIFICATION
# ==========================================

def is_broad_question(question):

    q = question.lower()

    broad_keywords = [
        "who are",
        "who was",
        "who were",
        "who is",
        "all employees",
        "all people",
        "everyone",
        "list all",
        "show all",
        "how many",
        "which employees",
        "which people",
        "summarize everyone",
        "summarize all"
    ]

    return any(
        keyword in q
        for keyword in broad_keywords
    )


# ==========================================
# RETRIEVE DOCUMENTS
# ==========================================

def retrieve_documents(question, vector_db):

    if is_broad_question(question):

        search_query = question

        if "leave" in question.lower():
            search_query = "LEAVE employee name date"

        results = vector_db.similarity_search(
            search_query,
            k=BROAD_TOP_K
        )

    else:

        results = vector_db.similarity_search(
            question,
            k=NORMAL_TOP_K
        )

    unique_results = {}

    for doc in results:

        key = (
            doc.page_content,
            doc.metadata.get("page")
        )

        unique_results[key] = doc

    return list(unique_results.values())


# ==========================================
# BUILD PROMPT
# ==========================================

def build_prompt(question, documents):

    context_parts = []

    for doc in documents:

        page = doc.metadata.get("page", 0) + 1

        context_parts.append(
            f"""
SOURCE PAGE: {page}

{doc.page_content}
"""
        )

    context = "\n\n".join(context_parts)

    return f"""
You are a reliable PDF question-answering assistant.

Answer the user's question using only the supplied PDF content.

PDF CONTEXT:
{context}

USER QUESTION:
{question}

RULES:

1. Use only the information in the PDF.
2. Match names, dates and facts carefully.
3. Never invent information.
4. If the user asks WHO, WHICH or ALL, include
   every matching result available in the context.
5. If the user asks HOW MANY, count only supported records.
6. Match exact dates for date-specific questions.
7. Distinguish WORKED, LEAVE and CLIENT_HOLIDAY.
8. A missing date does not mean leave or absence.
9. If information is unavailable, say so clearly.
10. Keep the answer concise.

Return only the final answer.
"""


# ==========================================
# GENERATE ANSWER
# ==========================================

def generate_answer(llm, prompt):

    for attempt in range(3):

        try:

            response = llm.invoke(prompt)
            content = response.content

            if isinstance(content, str):
                return content.strip()

            if isinstance(content, list):

                return "\n".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict)
                    and block.get("type") == "text"
                ).strip()

            return str(content)

        except Exception as error:

            message = str(error).lower()

            retryable = any(
                word in message
                for word in [
                    "429",
                    "503",
                    "unavailable",
                    "resource_exhausted",
                    "rate limit"
                ]
            )

            if not retryable or attempt == 2:
                raise

            time.sleep(10 * (attempt + 1))


# ==========================================
# MAIN QUESTION ANSWERING FUNCTION
# ==========================================

def ask_question(question, vector_db, llm):

    documents = retrieve_documents(
        question,
        vector_db
    )

    if not documents:
        return "I could not find relevant information in the PDF."

    prompt = build_prompt(
        question,
        documents
    )

    return generate_answer(llm, prompt)
