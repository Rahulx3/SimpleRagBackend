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
    Retrieves relevant PDF content using vector search and generates an answer using the LLM.
    Uses `search_query` for retrieval if provided (useful for contextualized follow-up questions).
    """
    query_to_search = search_query.strip() if search_query and search_query.strip() else question

    logger.info("Performing vector search for query: '%s'", query_to_search)

    # 1. Retrieve relevant documents
    documents = vector_db.similarity_search(query_to_search, k=4)

    if not documents:
        return "I couldn't find relevant information in the document to answer your question."

    # 2. Prepare PDF context
    context = "\n\n".join(doc.page_content for doc in documents)

    # 3. Prepare conversation history
    conversation = format_history(history, max_messages=5)

    # 4. Create prompt
    prompt = f"""You are a document-based question-answering assistant.

Answer the user's question using the provided context from the document.

Previous conversation:
{conversation}

Current question:
{question}

Document Context:
{context}

Instructions:
- Use document context for factual answers.
- Use conversation history to understand follow-up questions.
- Do not invent or hallucinate information not present in the context.
- If information is missing or unclear, clearly state that it is not available.
- Do not treat missing dates as leave or absence unless explicitly stated.
- Provide a clear, concise, and professional answer.
"""

    # 5. Generate and extract answer
    response = llm.invoke(prompt)
    return extract_text_content(response)
