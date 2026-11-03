import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from services.utils import format_history, extract_text_content

logger = logging.getLogger(__name__)


class QueryRoute(str, Enum):
    GENERAL = "GENERAL"
    DOCUMENT = "DOCUMENT"
    CLARIFICATION = "CLARIFICATION"


@dataclass
class RouteResult:
    route: QueryRoute
    search_query: str


def classify_query(question: str, history: list[Any], llm: Any) -> RouteResult:
    """
    Classifies the user's question into GENERAL, DOCUMENT, or CLARIFICATION.
    If the question is DOCUMENT and depends on previous conversation context,
    it also provides a standalone rewritten search query for vector retrieval.
    """
    conversation = format_history(history, max_messages=5)

    prompt = f"""You are an intent classification and query rewriter for an AI assistant.

The assistant has access to a PDF containing employee timesheets, attendance, leave records, and daily work details.

Your job:
1. Classify the user question into exactly ONE category:
   - GENERAL: Greetings, casual chat, general knowledge, or topics unrelated to the PDF.
   - DOCUMENT: Questions about employees, attendance, leave, timesheets, working hours, dates, daily tasks, or follow-ups to PDF answers.
   - CLARIFICATION: The user's query is ambiguous or too incomplete to determine intent.

2. If the category is DOCUMENT, provide a STANDALONE search query for vector database retrieval.
   - If the question is a follow-up (e.g. "What about him?", "How many leaves did he take?"), rewrite it using context from previous conversation to include specific names, subjects, and dates.
   - If the question is already self-contained, keep the search query identical to the current question.

Previous conversation:
{conversation}

Current question:
{question}

Respond ONLY in the following format:
CATEGORY: <GENERAL | DOCUMENT | CLARIFICATION>
SEARCH_QUERY: <standalone search query>
"""

    fallback = RouteResult(route=QueryRoute.DOCUMENT, search_query=question)

    try:
        response = llm.invoke(prompt)
        content = extract_text_content(response)

        # Parse category
        category_match = re.search(r"CATEGORY:\s*([A-Z]+)", content, re.IGNORECASE)
        search_query_match = re.search(r"SEARCH_QUERY:\s*(.+)", content, re.IGNORECASE)

        detected_category = category_match.group(1).upper() if category_match else ""
        rewritten_query = (
            search_query_match.group(1).strip()
            if search_query_match and search_query_match.group(1).strip()
            else question
        )

        for route in QueryRoute:
            if detected_category == route.value:
                return RouteResult(route=route, search_query=rewritten_query)

        # Fallback keyword match if model didn't strictly follow format
        first_line = content.split("\n")[0].upper()
        for route in QueryRoute:
            if route.value in first_line:
                return RouteResult(route=route, search_query=rewritten_query)

        logger.warning("Could not match category from router response: %s", content)
        return fallback

    except Exception:
        logger.exception("Query router classification failed; falling back to DOCUMENT.")
        return fallback
