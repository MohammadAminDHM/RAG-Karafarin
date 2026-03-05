from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.services.reranker import rerank_candidates

META_PATTERNS = [
    r"این\s*فایل",
    r"فایل\s*درباره",
    r"درباره\s*چی",
    r"درباره\s*چیه",
    r"jsonl",
    r"dataset|دیتاست",
    r"خلاصه|summary",
    r"ساختار|structure",
    r"چه\s*موضوعاتی",
    r"چه\s*چیزهایی\s*داخل",
]

def is_meta_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    for pat in META_PATTERNS:
        if re.search(pat, q, flags=re.IGNORECASE):
            return True
    return False


def choose_best_answer(
    query: str,
    results: List[Dict[str, Any]],
    min_vector_score: float = 0.22,
    min_overlap: int = 0,  # kept for backward compat (not used now)
    rerank_enabled: bool = True,
    min_combined: float = 0.30,
    alpha: float = 0.60,
    beta: float = 0.25,
    gamma: float = 0.15,
) -> Dict[str, Any]:
    if not results:
        return {"ok": False, "reason": "no_results"}

    # Rerank
    ranked = rerank_candidates(
        query=query,
        results=results,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    ) if rerank_enabled else results

    best = ranked[0]
    v = float(best.get("vector_score", best.get("score", 0.0)) or 0.0)
    combined = float(best.get("combined_score", v) or 0.0)
    ans = (best.get("answer_extracted") or "").strip()

    # Hard rules:
    if v < float(min_vector_score):
        return {"ok": False, "reason": "low_vector_score", "best": {"vector_score": v, "combined": combined}}

    if combined < float(min_combined):
        return {"ok": False, "reason": "low_combined", "best": {"vector_score": v, "combined": combined}}

    if not ans:
        return {"ok": False, "reason": "empty_answer", "best": {"vector_score": v, "combined": combined}}

    return {
        "ok": True,
        "answer": ans,
        "chosen": {
            "chunk_id": best.get("chunk_id"),
            "doc_id": best.get("doc_id"),
            "record_index": best.get("record_index"),
            "vector_score": v,
            "combined_score": combined,
            "match_char": best.get("match_char"),
            "match_jaccard": best.get("match_jaccard"),
            "question_extracted": best.get("question_extracted"),
        },
        "ranked_preview": [
            {
                "chunk_id": r.get("chunk_id"),
                "doc_id": r.get("doc_id"),
                "record_index": r.get("record_index"),
                "vector_score": float(r.get("vector_score", r.get("score", 0.0)) or 0.0),
                "combined_score": float(r.get("combined_score", r.get("score", 0.0)) or 0.0),
                "question_extracted": r.get("question_extracted"),
                "answer_extracted": (r.get("answer_extracted") or "")[:120],
            }
            for r in ranked[:5]
        ],
    }
