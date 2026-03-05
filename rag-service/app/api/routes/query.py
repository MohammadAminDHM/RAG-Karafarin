import time
from fastapi import APIRouter, Depends, Request

from app.api.deps import get_pipeline, get_generator_router
from app.core.config import get_settings
from app.rag.pipeline import RAGPipeline
from app.providers.router import GeneratorRouter
from app.schemas.query import QueryRequest, QueryResponse, QueryResponseMeta
from app.services.qa_answering import is_meta_query, choose_best_answer

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query_endpoint(
    request: Request,
    payload: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
    gen_router: GeneratorRouter = Depends(get_generator_router),
) -> QueryResponse:
    settings = get_settings()
    started = time.perf_counter()
    rid = getattr(request.state, "request_id", None)

    metrics = getattr(request.app.state, "metrics", None)
    if metrics:
        metrics.inc("queries_total", 1)

    top_k = min(payload.top_k, settings.max_top_k)

    # Pull more candidates for rerank/extract (no reindex)
    candidate_k = int(getattr(settings, "qa_candidate_k", 60) or 60)
    candidate_k = max(candidate_k, top_k)

    retrieved = pipeline.retrieve(query=payload.query, top_k=candidate_k)

    # Meta question: avoid returning a random support reply
    if is_meta_query(payload.query):
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return QueryResponse(
            answer=(
                "این فایل یک دیتاست پرسش/پاسخ (FAQ/Support) است. "
                "برای پاسخ درست، سوال را مثل سوال مشتری مطرح کن (مثلاً «رمز پویا نمیاد»)."
            ),
            meta=QueryResponseMeta(
                request_id=rid,
                provider_used="meta_rule",
                fallback_reason=None,
                retrieval_count=retrieved["retrieval_count"],
                latency_ms=elapsed_ms,
            ),
            sources=retrieved["sources"][:top_k],
        )

    # QA-mode extraction + rerank
    qa_mode = bool(getattr(settings, "qa_mode", True))
    if qa_mode:
        picked = choose_best_answer(
            query=payload.query,
            results=retrieved["raw_results"],
            min_vector_score=float(getattr(settings, "qa_min_score", 0.22)),
            min_overlap=int(getattr(settings, "qa_min_overlap", 0)),
            rerank_enabled=bool(getattr(settings, "rerank_enabled", True)),
            min_combined=float(getattr(settings, "qa_min_combined", 0.30)),
            alpha=float(getattr(settings, "rerank_alpha", 0.60)),
            beta=float(getattr(settings, "rerank_beta", 0.25)),
            gamma=float(getattr(settings, "rerank_gamma", 0.15)),
        )

        elapsed_ms = int((time.perf_counter() - started) * 1000)

        if picked.get("ok"):
            return QueryResponse(
                answer=picked["answer"],
                meta=QueryResponseMeta(
                    request_id=rid,
                    provider_used="qa_extract_rerank",
                    fallback_reason=None,
                    retrieval_count=retrieved["retrieval_count"],
                    latency_ms=elapsed_ms,
                ),
                sources=retrieved["sources"][:top_k],
            )

        # Not confident: show top ranked previews as guidance
        reason = str(picked.get("reason"))
        ranked_preview = picked.get("ranked_preview") or []
        return QueryResponse(
            answer=(
                f"به پاسخ مطمئن نرسیدم ({reason}). "
                "چند مورد خیلی نزدیک پیدا کردم؛ اگر یکی همون موضوعه، با جزئیات‌تر بگو."
            ),
            meta=QueryResponseMeta(
                request_id=rid,
                provider_used="qa_extract_rerank",
                fallback_reason=reason,
                retrieval_count=retrieved["retrieval_count"],
                latency_ms=elapsed_ms,
            ),
            sources=ranked_preview,
        )

    # If QA_MODE=false: use LLM generation local->api (kept for later)
    system = (
        "You are a QA assistant. The CONTEXT contains retrieved QA entries.\n"
        "Answer using ONLY the CONTEXT. If not found, say you don't know."
    )
    user = f"USER QUESTION:\n{payload.query}\n\nCONTEXT:\n{retrieved['context']}"
    gen = gen_router.generate(system=system, user=user)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return QueryResponse(
        answer=gen.answer,
        meta=QueryResponseMeta(
            request_id=rid,
            provider_used=gen.provider_used,
            fallback_reason=gen.fallback_reason,
            retrieval_count=retrieved["retrieval_count"],
            latency_ms=elapsed_ms,
        ),
        sources=retrieved["sources"][:top_k],
    )
