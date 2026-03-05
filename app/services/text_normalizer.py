from __future__ import annotations
import re

_ARABIC_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
_TATWEEL = "\u0640"
_ZWNJ = "\u200c"
_PUNCT = re.compile(r"[^\w\s\u0600-\u06FF]+", flags=re.UNICODE)
_MULTI_SPACE = re.compile(r"\s+")

def normalize_chars_fa(text: str) -> str:
    if not text:
        return ""
    t = text
    t = t.replace("ي", "ی").replace("ك", "ک")
    t = t.replace("ة", "ه")
    t = t.replace(_ZWNJ, " ")
    t = t.replace(_TATWEEL, "")
    t = _ARABIC_DIACRITICS.sub("", t)
    t = _MULTI_SPACE.sub(" ", t).strip()
    return t


def normalize_for_match(text: str) -> str:
    """
    Stronger normalization for lexical similarity and reranking.
    Keeps Persian letters and word chars, strips punctuation noise.
    """
    t = normalize_chars_fa(text).lower()
    t = _PUNCT.sub(" ", t)
    t = _MULTI_SPACE.sub(" ", t).strip()
    return t
