from __future__ import annotations
import json
from pathlib import Path
from fastapi import APIRouter, Request

from app.core.config import get_settings
from app.schemas.common import StatusResponse

router = APIRouter(tags=["status"])


def _safe_config(settings) -> dict:
    return {
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "default_top_k": settings.default_top_k,
        "max_top_k": settings.max_top_k,
        "max_context_chars": settings.max_context_chars,
        "static_source_path": settings.static_source_path,
        "ollama_base_url": settings.ollama_base_url,
        "ollama_embed_model": settings.ollama_embed_model,
        "ollama_chat_model": settings.ollama_chat_model,
        "max_local_concurrent": settings.max_local_concurrent,
        "api_fallback_enabled": settings.api_fallback_enabled,
        "api_base_url": settings.api_base_url,
        "api_chat_path": settings.api_chat_path,
        "api_model": settings.api_model,
        "faiss_index_path": settings.faiss_index_path,
        "faiss_metadata_path": settings.faiss_metadata_path,
        "index_state_path": settings.index_state_path,
        "embedding_dim": settings.embedding_dim,
        "auto_ingest_on_startup": settings.auto_ingest_on_startup,
    }


def _read_index_state(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


@router.get("/status", response_model=StatusResponse)
def status(request: Request) -> StatusResponse:
    settings = get_settings()

    pipeline_ready = getattr(request.app.state, "rag_pipeline", None) is not None
    ingestion_report = getattr(request.app.state, "ingestion_report", {}) or {}
    index_state = _read_index_state(settings.index_state_path)

    return StatusResponse(
        service=settings.app_name,
        env=settings.app_env,
        version=settings.app_version,
        api_prefix=settings.api_prefix,
        rag_ready=pipeline_ready,
        ingestion_report=ingestion_report,
        index_state=index_state,
        config_safe=_safe_config(settings),
    )
