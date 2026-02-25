import re


def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - Normalize line endings
    - Collapse extra spaces
    - Keep semantics
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
