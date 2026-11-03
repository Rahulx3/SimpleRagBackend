import logging
from typing import Any

from services.utils import format_history, extract_text_content

logger = logging.getLogger(__name__)


def generate_document_answer(
    question: str,
    history: list[Any],
    vector_db: Any,
    llm: Any,
    search_query: str | None = None
) -> str:
    """
    Retrieves relevant document content using vector search and generates an answer using the LLM.
    Uses `search_query` for retrieval if provided (useful for contextualized follow-up questions).
    """
    query_to_search = search_query.strip() if search_query and search_query.strip() else question

    logger.info("Performing vector search for query: '%s'", query_to_search)

    # 1. Retrieve relevant documents (k=6 for comprehensive multi-day/employee coverage)
    documents = vector_db.similarity_search(query_to_search, k=6)

    if not documents:
        return "I couldn't find relevant information in the document to answer your question."

    # 2. Prepare PDF context
    context = "\n\n".join(doc.page_content for doc in documents)

    # 3. Prepare conversation history
    conversation = format_history(history, max_messages=5)

    # 4. Create prompt
    prompt = f"""You are a document-based question-answering assistant for an organization's timesheet and work activity records.

Answer the user's question accurately using ONLY the provided context from the document.

Previous conversation:
{conversation}

Current question:
{question}

Document Context:
{context}

Instructions:
- Carefully search the context for the exact employee name and date mentioned in the question.
- The document has both monthly attendance summaries and day-by-day activity logs.
- For date-specific queries (e.g. "19 October 2026"):
  - Locate the exact date in the employee's work log or dated section.
  - State their Status (e.g. WORKED, LEAVE, CLIENT_HOLIDAY), Hours logged, and exact Task / Activity.
- If the user asks about a weekend date (Saturday or Sunday) that is not in the working days log, explain that it was a weekend and mention what they did on the surrounding Friday or Monday.
- If an employee was on leave or client holiday, mention their leave code or holiday note if present.
- Provide a clear, direct, and factually accurate answer.
"""

    # 5. Generate and extract answer
    response = llm.invoke(prompt)
    return extract_text_content(response)
