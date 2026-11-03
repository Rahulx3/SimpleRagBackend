from typing import Any


def format_history(history: list[Any], max_messages: int = 5) -> str:
    """
    Formats the conversation history into a clean string for prompts.
    Handles both Pydantic models (with .role, .content) and dicts.
    """
    if not history:
        return "None"

    formatted_lines: list[str] = []
    for message in history[-max_messages:]:
        role = getattr(message, "role", None) or (message.get("role") if isinstance(message, dict) else "user")
        content = getattr(message, "content", None) or (message.get("content") if isinstance(message, dict) else "")

        formatted_lines.append(f"{role.capitalize()}: {content.strip()}")

    return "\n".join(formatted_lines) if formatted_lines else "None"


def extract_text_content(response: Any) -> str:
    """
    Extracts plain text content from LangChain LLM response objects.
    """
    if not response:
        return ""

    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return " ".join(text_parts).strip()

    return str(content).strip()
