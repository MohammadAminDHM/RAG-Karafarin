from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

from app.services.text_normalizer import normalize_for_match

STOPWORDS = {
    "و","یا","به","از","در","را","با","برای","این","آن","یک","که","تا","هم","اما","اگر","پس","می","شود","شده","کرد","کردن","کرده",
    "the","a","an","to","of","in","on","and","or","is","are","was","were","be","for","with","as","it","this","that"
}

Q_RE = re.compile(r"^\s*question\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
A_RE = re.compile(r"^\s*answ\w*\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)

def parse_qa(text: str) -> Tuple[str, str]:
    if not text:
        return "", ""
    qm = Q_RE.search(text)
    am = A_RE.search(text)
    q = qm.group(1).strip() if qm else ""
    a = am.group(1).strip() if am else ""
    return q, a

def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]

def jaccard(a_tokens: List[str], b_tokens: List[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    a, b = set(a_tokens), set(b_tokens)
    inter = len(a & b)
    uni = len(a | b) or 1
    return inter / uni

def char_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def rerank_candidates(
    query: str,
    results: List[Dict[str, Any]],
    alpha: float = 0.60,
    beta: float = 0.25,
    gamma: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    Returns candidates with extra fields:
      - question_extracted
      - answer_extracted
      - match_char
      - match_jaccard
      - combined_score
    Sorted descending by combined_score.
    """
    q_norm = normalize_for_match(query)
    q_tokens = _tokenize(q_norm)

    ranked: List[Dict[str, Any]] = []
    for r in results:
        raw = (r.get("text") or "").strip()
        cand_q, cand_a = parse_qa(raw)
        cand_q_norm = normalize_for_match(cand_q)
        cand_tokens = _tokenize(cand_q_norm)

        vscore = float(r.get("score", 0.0) or 0.0)
        cscore = char_similarity(q_norm, cand_q_norm)
        jscore = jaccard(q_tokens, cand_tokens)

        combined = alpha * vscore + beta * cscore + gamma * jscore

        ranked.append({
            **r,
            "question_extracted": cand_q,
            "answer_extracted": cand_a,
            "match_char": float(cscore),
            "match_jaccard": float(jscore),
            "combined_score": float(combined),
            "vector_score": float(vscore),
        })

    ranked.sort(key=lambda x: x["combined_score"], reverse=True)
    return ranked
