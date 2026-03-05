import logging
import threading

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes.health import router as health_router
from app.api.routes.query import router as query_router
from app.api.routes.status import router as status_router
from app.api.routes.admin import router as admin_router
from app.api.routes.corpus import router as corpus_router

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.exceptions import AppError
from app.middleware.request_id import RequestIDMiddleware
from app.middleware.timing import TimingMiddleware
from app.schemas.common import ErrorResponse
from app.services.metrics_service import Metrics
from app.services.ingestion_service import build_pipeline_from_existing_index, rebuild_index_and_pipeline

from app.services.busy_detector import BusyDetector, CircuitBreaker
from app.providers.local_provider import OllamaChatProvider
from app.providers.api_provider import OpenAICompatChatProvider
from app.providers.router import GeneratorRouter

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
)

# Middlewares
app.add_middleware(RequestIDMiddleware)
app.add_middleware(TimingMiddleware)

# App state
app.state.metrics = Metrics()
app.state.reindex_lock = threading.Lock()
app.state.rag_pipeline = None
app.state.ingestion_report = {"indexed": False, "reason": "startup_not_run"}

# Generation router state
app.state.generator_router = None

# Corpus summary cache (Phase 3.1)
app.state.corpus_summary_cache = None

# Routers
app.include_router(health_router, prefix=settings.api_prefix)
app.include_router(status_router, prefix=settings.api_prefix)
app.include_router(corpus_router, prefix=settings.api_prefix)
app.include_router(query_router, prefix=settings.api_prefix)
app.include_router(admin_router, prefix=settings.api_prefix)


@app.on_event("startup")
def on_startup() -> None:
    """
    Startup:
    1) Setup generator router (local busy -> api fallback)
    2) Setup RAG retrieval pipeline (load index or rebuild)
    """
    try:
        # 1) Generator Router
        busy = BusyDetector(max_concurrent=settings.max_local_concurrent)
        circuit = CircuitBreaker(
            fails_to_open=settings.local_fails_to_open_circuit,
            reset_sec=settings.local_circuit_reset_sec,
        )

        local = OllamaChatProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_chat_model,
            timeout_sec=settings.ollama_chat_timeout_sec,
            temperature=settings.ollama_chat_temperature,
            top_p=settings.ollama_chat_top_p,
            top_k=settings.ollama_chat_top_k,
            repeat_penalty=settings.ollama_chat_repeat_penalty,
            num_ctx=settings.ollama_chat_num_ctx,
            max_tokens=settings.ollama_chat_max_tokens,
        )

        api = None
        if settings.api_fallback_enabled:
            api = OpenAICompatChatProvider(
                base_url=settings.api_base_url,
                chat_path=settings.api_chat_path,
                api_key=settings.api_key,
                model=settings.api_model,
                timeout_sec=settings.api_timeout_sec,
                max_retries=settings.api_max_retries,
                temperature=settings.api_temperature,
                max_tokens=settings.api_max_tokens,
                top_p=settings.api_top_p,
            )

        app.state.generator_router = GeneratorRouter(
            local=local,
            api=api,
            busy=busy,
            circuit=circuit,
            api_fallback_enabled=settings.api_fallback_enabled,
        )

        # 2) RAG pipeline
        pipeline, report = build_pipeline_from_existing_index(settings)
        if pipeline is not None:
            app.state.rag_pipeline = pipeline
            app.state.ingestion_report = report
            logger.info("RAG ready (loaded): %s", report)
            return

        if settings.auto_ingest_on_startup:
            pipeline, report = rebuild_index_and_pipeline(settings)
            app.state.rag_pipeline = pipeline
            app.state.ingestion_report = report
            logger.info("RAG ready (rebuilt): %s", report)

    except Exception as exc:
        logger.exception("Startup failed: %s", exc)
        app.state.ingestion_report = {"indexed": False, "error": str(exc)}


# ---- Exception handlers (standard error shape) ----

@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    rid = getattr(request.state, "request_id", None)
    app.state.metrics.inc("requests_error", 1)
    payload = ErrorResponse(
        error=exc.message,
        error_code=exc.code,
        request_id=rid,
        details=exc.details or None,
    )
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    rid = getattr(request.state, "request_id", None)
    app.state.metrics.inc("requests_error", 1)
    payload = ErrorResponse(
        error=str(exc.detail),
        error_code="http_error",
        request_id=rid,
        details={"status_code": exc.status_code},
    )
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", None)
    app.state.metrics.inc("requests_error", 1)
    payload = ErrorResponse(
        error="Internal server error",
        error_code="internal_error",
        request_id=rid,
        details=None,
    )
    logger.exception("Unhandled error (request_id=%s): %s", rid, exc)
    return JSONResponse(status_code=500, content=payload.model_dump())


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    app.state.metrics.inc("requests_total", 1)
    try:
        response = await call_next(request)
        if 200 <= response.status_code < 400:
            app.state.metrics.inc("requests_ok", 1)
        else:
            app.state.metrics.inc("requests_error", 1)
        return response
    except Exception:
        app.state.metrics.inc("requests_error", 1)
        raise


@app.get("/")
def root() -> dict:
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "api_prefix": settings.api_prefix,
    }


@app.get("/metrics")
def metrics() -> dict:
    return app.state.metrics.snapshot()
