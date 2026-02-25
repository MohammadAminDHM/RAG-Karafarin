from fastapi import HTTPException, Request

from app.core.config import get_settings
from app.rag.pipeline import RAGPipeline
from app.services.ingestion_service import build_pipeline_from_existing_index, rebuild_index_and_pipeline


def get_pipeline(request: Request) -> RAGPipeline:
    pipeline = getattr(request.app.state, "rag_pipeline", None)
    if pipeline is not None:
        return pipeline

    # If app restarted / worker fresh: try fast load from disk
    settings = get_settings()
    pipeline, report = build_pipeline_from_existing_index(settings)
    if pipeline is not None:
        request.app.state.rag_pipeline = pipeline
        request.app.state.ingestion_report = report
        return pipeline

    # Optional: as last resort, rebuild (only if allowed)
    if settings.auto_ingest_on_startup:
        pipeline, report = rebuild_index_and_pipeline(settings)
        request.app.state.rag_pipeline = pipeline
        request.app.state.ingestion_report = report
        return pipeline

    raise HTTPException(status_code=503, detail="RAG pipeline is not ready (index missing)")
