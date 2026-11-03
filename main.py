from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag import initialize_models, initialize_vector_db
from routers.chat import router as chat_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("rag_api")


# APPLICATION STATE
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Gemini models and ChromaDB vector database...")

    embeddings, llm = initialize_models()
    vector_db = initialize_vector_db(embeddings)

    app.state.llm = llm
    app.state.vector_db = vector_db

    logger.info("RAG application initialized successfully and ready for traffic.")

    yield

    app.state.llm = None
    app.state.vector_db = None

    logger.info("Application shutdown completed.")


# FASTAPI APP
app = FastAPI(
    title="PDF RAG API",
    description="Router-based PDF question answering application",
    version="2.0.0",
    lifespan=lifespan
)


# CORS CONFIGURATION
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://rag-hazel-six.vercel.app",
        "https://rag-38x7n0w74-rahul-sajwans-projects.vercel.app",
        "https://rag-38x7n0w74-rahul-sajwans-projects.vercel.app/",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ROUTERS
app.include_router(chat_router)


# HEALTH ENDPOINT
@app.get("/")
def health_check():
    return {
        "message": "PDF RAG API is running",
        "status": "healthy"
    }
