import time
from fastapi import APIRouter
from app.schemas.query import QueryRequest, QueryResponse, QueryResponseMeta

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query_endpoint(payload: QueryRequest) -> QueryResponse:
    """
    Phase 0 stub endpoint.
    Real RAG pipeline will be connected in Phase 1.
    """
    started = time.perf_counter()

    # Stub response to verify request/response flow
    answer = (
        "Phase 0 is active ✅ FastAPI + schemas + config are working. "
        "RAG pipeline will be connected in Phase 1."
    )

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    return QueryResponse(
        answer=answer,
        meta=QueryResponseMeta(
            provider_used="stub",
            retrieval_count=0,
            latency_ms=elapsed_ms,
        ),
        sources=[],
    )
