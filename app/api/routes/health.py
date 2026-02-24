from fastapi import APIRouter
from app.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "service": settings.app_name,
        "env": settings.app_env,
        "version": settings.app_version,
    }


@router.get("/ready")
def ready() -> dict:
    # In later phases we will check vector DB, model connectivity, etc.
    return {
        "status": "ready",
        "checks": {
            "api": True,
            "vectorstore": "not_checked_yet",
            "local_provider": "not_checked_yet",
            "api_provider": "not_checked_yet",
        },
    }
