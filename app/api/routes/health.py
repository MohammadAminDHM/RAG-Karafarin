from fastapi import APIRouter, Request
from app.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request) -> dict:
    settings = get_settings()
    rid = getattr(request.state, "request_id", None)

    return {
        "status": "ok",
        "service": settings.app_name,
        "env": settings.app_env,
        "version": settings.app_version,
        "request_id": rid,
    }


@router.get("/ready")
def ready(request: Request) -> dict:
    report = getattr(request.app.state, "ingestion_report", None)
    pipeline_ready = getattr(request.app.state, "rag_pipeline", None) is not None

    return {
        "status": "ready" if pipeline_ready else "not_ready",
        "checks": {
            "api": True,
            "rag_pipeline": pipeline_ready,
            "ingestion": report or "not_run",
            "local_provider": "not_connected_in_phase2",
            "api_provider": "not_connected_in_phase2",
        },
    }
