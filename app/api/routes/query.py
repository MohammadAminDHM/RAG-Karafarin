import time

from fastapi import APIRouter, Depends

from app.api.deps import get_pipeline
from app.core.config import get_settings
from app.rag.pipeline import RAGPipeline
from app.schemas.query import QueryRequest, QueryResponse, QueryResponseMeta

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query_endpoint(
    payload: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> QueryResponse:
    """
    Phase 1 RAG endpoint:
    - Query embedding via Ollama
    - Retrieval via FAISS
    - Heuristic answer (temporary until generation phase)
    """
    settings = get_settings()
    started = time.perf_counter()

    top_k = min(payload.top_k, settings.max_top_k)
    result = pipeline.ask(query=payload.query, top_k=top_k)

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    return QueryResponse(
        answer=result["answer"],
        meta=QueryResponseMeta(
            provider_used="phase1_rag_ollama_embed_faiss",
            retrieval_count=result["retrieval_count"],
            latency_ms=elapsed_ms,
        ),
        sources=result["sources"],
    )
