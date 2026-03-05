from __future__ import annotations

import re
from typing import Any, Dict, List

from app.services.reranker import rerank_candidates

META_PATTERNS = [
    r"این\s*فایل", r"درباره\s*چی", r"درباره\s*چیه",
    r"jsonl", r"dataset|دیتاست", r"خلاصه|summary", r"ساختار|structure",
]

def is_meta_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    return any(re.search(p, q, flags=re.IGNORECASE) for p in META_PATTERNS)


_LEADING_PHRASES = [
    r"^\s*با\s*سلام[،,:\-\s]+",
    r"^\s*سلام[،,:\-\s]+",
    r"^\s*مشتری\s*(?:گرامی|عزیز)[،,:\-\s]+",
    r"^\s*کاربر\s*(?:گرامی|عزیز)[،,:\-\s]+",
]
_MULTI_SPACE = re.compile(r"[ \t]+")
# Numbered list item: digit(s) followed by dot and space — break to new line
_NUMBERED_ITEM = re.compile(r"(?<!\n)(?<!\A)\s+(\d+\.\s)")
# Bullet dash at start of token (not inside words/paths like "a->b")
_BULLET_DASH = re.compile(r"(?<!\w)\s+-\s+(?=\S)")


def polish_answer_for_user(answer: str) -> str:
    """
    Formats a raw FAQ answer for clean user-facing display.
    - Strips repetitive greeting prefix
    - Formats numbered instructions as line-separated steps
    - Normalises whitespace
    - Does NOT change factual content
    """
    text = (answer or "").strip()
    if not text:
        return ""

    # Remove greeting/salutation prefix
    for pat in _LEADING_PHRASES:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    text = text.strip(" \t\n-,:")

    # Collapse inline whitespace (keep newlines already in source)
    lines = text.splitlines()
    lines = [_MULTI_SPACE.sub(" ", ln).strip() for ln in lines]
    text = "\n".join(ln for ln in lines if ln)

    # Break numbered steps to separate lines: "...بروید. 2. بعد..." → newline before "2."
    text = _NUMBERED_ITEM.sub(r"\n\1", text)

    # Bullets: " - item" → "\n• item" (only standalone dashes, not inside paths/arrows)
    text = _BULLET_DASH.sub("\n• ", text)

    # Ensure sentence ends with punctuation
    if text and not text[-1] in ("؟", ".", "!", "…", ":"):
        text += "."

    return text

def choose_best_answer(
    query: str,
    results: List[Dict[str, Any]],
    min_vector_score: float = 0.28,
    min_combined: float = 0.28,
    rerank_enabled: bool = True,
    alpha: float = 0.65,
    beta: float = 0.25,
    gamma: float = 0.10,
    # kept for call-site compat — no longer used as hard gates
    min_match_char: float = 0.0,
    min_match_jaccard: float = 0.0,
    min_margin_to_second: float = 0.0,
) -> Dict[str, Any]:
    if not results:
        return {"ok": False, "reason": "no_results"}

    ranked = rerank_candidates(query, results, alpha=alpha, beta=beta, gamma=gamma) if rerank_enabled else results
    best = ranked[0]
    v = float(best.get("vector_score", best.get("score", 0.0)) or 0.0)
    c = float(best.get("combined_score", v) or 0.0)
    ans = (best.get("answer") or "").strip()

    # Primary gate: vector embedding is the strongest signal for semantic RAG.
    # Using qwen3-embedding:8b a score >= min_vector_score reliably means
    # the question intent matches, even when the user words it differently.
    if v < float(min_vector_score):
        return {
            "ok": False,
            "reason": "low_vector_score",
            "best": {"vector_score": v, "combined": c},
            "ranked_preview": ranked[:5],
        }

    # Secondary gate: combined score (vector + lexical boost).
    if c < float(min_combined):
        return {
            "ok": False,
            "reason": "low_combined",
            "best": {"vector_score": v, "combined": c},
            "ranked_preview": ranked[:5],
        }

    if not ans:
        return {"ok": False, "reason": "empty_answer", "best": {"vector_score": v, "combined": c}}

    return {"ok": True, "answer": polish_answer_for_user(ans), "best": best, "ranked_preview": ranked[:5]}
