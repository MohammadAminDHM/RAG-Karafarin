from __future__ import annotations
import json
from pathlib import Path

from fastapi import APIRouter, Request
from app.core.config import get_settings
from app.core.exceptions import AppError
from app.services.ingestion_service import rebuild_index_and_pipeline, build_pipeline_from_existing_index

router = APIRouter(tags=["admin"])


@router.post("/admin/reindex")
def reindex(request: Request) -> dict:
    """
    Manual reindex endpoint.
    - Uses an app-level lock to prevent concurrent reindex
    """
    settings = get_settings()

    lock = getattr(request.app.state, "reindex_lock", None)
    if lock is None:
        raise AppError("Reindex lock not initialized", code="reindex_lock_missing", status_code=500)

    if not lock.acquire(blocking=False):
        raise AppError("Reindex already in progress", code="reindex_in_progress", status_code=409)

    try:
        pipeline, report = rebuild_index_and_pipeline(settings)
        request.app.state.rag_pipeline = pipeline
        request.app.state.ingestion_report = report

        metrics = getattr(request.app.state, "metrics", None)
        if metrics:
            metrics.inc("reindex_total", 1)

        return {"ok": True, "ingestion_report": report}
    finally:
        lock.release()


@router.post("/admin/reload-index")
def reload_index(request: Request) -> dict:
    """
    Reload from disk (fast path). Does NOT rebuild embeddings.
    Useful if you swapped index files.
    """
    settings = get_settings()
    pipeline, report = build_pipeline_from_existing_index(settings)
    if pipeline is None:
        raise AppError("Index not available to load", code="index_load_failed", status_code=404, details=report)

    request.app.state.rag_pipeline = pipeline
    request.app.state.ingestion_report = report
    return {"ok": True, "ingestion_report": report}


@router.get("/admin/index-state")
def index_state() -> dict:
    settings = get_settings()
    p = Path(settings.index_state_path)
    if not p.exists():
        return {"exists": False, "state": {}}
    try:
        return {"exists": True, "state": json.loads(p.read_text(encoding="utf-8"))}
    except Exception as exc:
        raise AppError("Failed to read index state", code="index_state_read_failed", status_code=500, details={"error": str(exc)})
