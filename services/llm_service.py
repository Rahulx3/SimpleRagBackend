import logging
from typing import Any

from services.utils import format_history, extract_text_content

logger = logging.getLogger(__name__)


def generate_general_answer(question: str, history: list[Any], llm: Any) -> str:
    """
    Generates an answer using LLM knowledge and conversation history without PDF retrieval.
    """
    conversation = format_history(history, max_messages=5)

    prompt = f"""You are a helpful AI assistant.

Answer general questions naturally and conversationally.

Previous conversation:
{conversation}

Current question:
{question}

Instructions:
- Use conversation history when relevant.
- Answer clearly and directly.
- Do not claim to have searched any document or PDF.
- If the user asks a general knowledge question, answer using your knowledge.
"""

    response = llm.invoke(prompt)
    return extract_text_content(response)
