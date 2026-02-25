import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.query import router as query_router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.services.ingestion_service import build_pipeline_from_existing_index, rebuild_index_and_pipeline

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s (%s)", settings.app_name, settings.app_env)

    app.state.rag_pipeline = None
    app.state.ingestion_report = {"indexed": False, "reason": "startup_not_run"}

    # 1) Fast path: load existing index if possible
    pipeline, report = build_pipeline_from_existing_index(settings)
    if pipeline is not None:
        app.state.rag_pipeline = pipeline
        app.state.ingestion_report = report
        logger.info("RAG ready (loaded): %s", report)
        yield
        logger.info("Shutting down %s", settings.app_name)
        return

    # 2) Slow path: rebuild only if allowed
    if settings.auto_ingest_on_startup:
        try:
            pipeline, report = rebuild_index_and_pipeline(settings)
            app.state.rag_pipeline = pipeline
            app.state.ingestion_report = report
            logger.info("RAG ready (rebuilt): %s", report)
        except Exception as exc:
            logger.exception("RAG rebuild failed: %s", exc)
            app.state.ingestion_report = {"indexed": False, "error": str(exc)}

    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
)

app.include_router(health_router, prefix=settings.api_prefix)
app.include_router(query_router, prefix=settings.api_prefix)


@app.get("/")
def root() -> dict:
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "api_prefix": settings.api_prefix,
    }
