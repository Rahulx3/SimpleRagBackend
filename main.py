
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag import (
    initialize_models,
    initialize_vector_db,
    ask_question
)


# ==========================================
# APPLICATION STATE
# ==========================================

app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):

    embeddings, llm = initialize_models()

    vector_db = initialize_vector_db(embeddings)

    app_state["vector_db"] = vector_db
    app_state["llm"] = llm

    yield

    app_state.clear()


# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI(
    title="PDF RAG API",
    description="Ask questions from uploaded PDF knowledge.",
    version="1.0.0",
    lifespan=lifespan
)


# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ==========================================
# REQUEST MODEL
# ==========================================

class ChatRequest(BaseModel):

    question: str = Field(
        min_length=1,
        max_length=2000
    )


# ==========================================
# RESPONSE MODEL
# ==========================================

class ChatResponse(BaseModel):

    answer: str


# ==========================================
# HEALTH API
# ==========================================

@app.get("/")
def home():

    return {
        "message": "PDF RAG API is running"
    }


# ==========================================
# CHAT API
# ==========================================

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    try:

        answer = ask_question(
            request.question,
            app_state["vector_db"],
            app_state["llm"]
        )

        return ChatResponse(
            answer=answer
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unable to process your question."
        )
