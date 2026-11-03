import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from services.llm_service import generate_general_answer
from services.query_router import QueryRoute, classify_query
from services.rag_service import generate_document_answer

logger = logging.getLogger(__name__)

router = APIRouter()


# ==========================================
# REQUEST & RESPONSE MODELS
# ==========================================

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=10)


class ChatResponse(BaseModel):
    answer: str
    route: str


# ==========================================
# CHAT API
# ==========================================

@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, app_request: Request):
    """
    Classifies the user question and routes to either general LLM answering,
    RAG document retrieval, or clarification.
    """
    try:
        llm = app_request.app.state.llm
        vector_db = app_request.app.state.vector_db

        if not llm or not vector_db:
            raise RuntimeError("Application models or vector database are not initialized.")

        # Step 1: Classify question & extract standalone search query if applicable
        route_result = classify_query(
            question=request.question,
            history=request.history,
            llm=llm
        )

        route = route_result.route
        logger.info("Classified route: %s | Search Query: '%s'", route.value, route_result.search_query)

        # Step 2: Route execution
        if route == QueryRoute.GENERAL:
            answer = generate_general_answer(
                question=request.question,
                history=request.history,
                llm=llm
            )

        elif route == QueryRoute.DOCUMENT:
            answer = generate_document_answer(
                question=request.question,
                history=request.history,
                vector_db=vector_db,
                llm=llm,
                search_query=route_result.search_query
            )

        else:
            answer = "Could you please clarify your question?"

        return ChatResponse(
            answer=answer,
            route=route.value
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception("Chat processing failed for question: %s", request.question)
        raise HTTPException(
            status_code=500,
            detail="Unable to process your question at this time. Please try again later."
        )
