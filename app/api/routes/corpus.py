from __future__ import annotations

from fastapi import APIRouter, Request

from app.core.config import get_settings
from app.services.corpus_summary import compute_corpus_summary

router = APIRouter(tags=["corpus"])


@router.get("/corpus/summary")
def corpus_summary(request: Request) -> dict:
    """
    Returns cached corpus summary if available; otherwise computes once.
    """
    cached = getattr(request.app.state, "corpus_summary_cache", None)
    if isinstance(cached, dict) and cached.get("ok"):
        return {"cached": True, "summary": cached}

    settings = get_settings()
    summary = compute_corpus_summary(
        faiss_metadata_path=settings.faiss_metadata_path,
        index_state_path=settings.index_state_path,
    )

    request.app.state.corpus_summary_cache = summary
    return {"cached": False, "summary": summary}
