import logging
import time
from fastapi import APIRouter, Depends, Request

from app.api.deps import get_pipeline, get_generator_router
from app.core.config import get_settings
from app.rag.pipeline import RAGPipeline
from app.providers.router import GeneratorRouter
from app.schemas.query import QueryRequest, QueryResponse, QueryResponseMeta
from app.services.qa_answering import is_meta_query, choose_best_answer

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])

_REASON_MESSAGES = {
    "no_results": "مورد مرتبطی در پایگاه دانش پیدا نکردم",
    "low_vector_score": "نتایج پیدا شده خیلی کم‌ارتباط بودند",
    "low_combined": "تطابق کافی برای پاسخ دقیق وجود نداشت",
    "low_lexical_match": "شباهت واژگانی کافی بین سوال شما و نتیجه برتر نبود",
    "ambiguous_top_match": "چند نتیجه نزدیک بودند و پاسخ یکتا نبود",
    "empty_answer": "برای نتیجه برتر پاسخ معتبری ثبت نشده بود",
}


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

    top_k = min(payload.top_k, settings.max_top_k)
    candidate_k = max(int(settings.qa_candidate_k), top_k)

    retrieved = pipeline.retrieve(query=payload.query, top_k=candidate_k)

    if is_meta_query(payload.query):
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return QueryResponse(
            answer="این فایل یک دیتاست پرسش/پاسخ (FAQ/Support) است. یک سوال مشخص مشتری بپرس تا جواب دقیق از دیتاست بدهم.",
            meta=QueryResponseMeta(
                request_id=rid,
                provider_used="meta_rule",
                fallback_reason=None,
                retrieval_count=retrieved["retrieval_count"],
                latency_ms=elapsed_ms,
            ),
            sources=retrieved["sources"][:top_k],
        )

    picked = choose_best_answer(
        query=payload.query,
        results=retrieved["raw_results"],
        min_vector_score=float(settings.qa_min_score),
        min_combined=float(settings.qa_min_combined),
        rerank_enabled=bool(settings.rerank_enabled),
        alpha=float(settings.rerank_alpha),
        beta=float(settings.rerank_beta),
        gamma=float(settings.rerank_gamma),
    )

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    if picked.get("ok"):
        best_meta = picked.get("best", {})
        logger.info(
            "query answered | rid=%s v=%.3f c=%.3f char=%.3f jac=%.3f",
            rid,
            float(best_meta.get("vector_score", 0)),
            float(best_meta.get("combined_score", 0)),
            float(best_meta.get("match_char", 0)),
            float(best_meta.get("match_jaccard", 0)),
        )
        return QueryResponse(
            answer=picked["answer"],
            meta=QueryResponseMeta(
                request_id=rid,
                provider_used="qa_extract_question_only",
                fallback_reason=None,
                retrieval_count=retrieved["retrieval_count"],
                latency_ms=elapsed_ms,
            ),
            sources=retrieved["sources"][:top_k],
        )

    reason_code = str(picked.get("reason") or "unknown")
    reason_text = _REASON_MESSAGES.get(reason_code, "اطمینان پاسخ پایین بود")
    best_dbg = picked.get("best", {})
    logger.warning(
        "query unanswered | rid=%s reason=%s best=%s",
        rid, reason_code, best_dbg,
    )
    # Expose closest matches so the user has useful hints
    ranked_preview = picked.get("ranked_preview") or []
    fallback_sources = [
        {
            "chunk_id": r.get("chunk_id"),
            "doc_id": r.get("doc_id"),
            "score": round(float(r.get("combined_score", r.get("score", 0))), 4),
            "text_preview": f"Q: {(r.get('question') or '')[:120]}",
        }
        for r in ranked_preview[:top_k]
    ]
    return QueryResponse(
        answer=f"به پاسخ مطمئن نرسیدم: {reason_text}. لطفاً سوال را دقیق‌تر و با جزئیات بیشتری بپرس.",
        meta=QueryResponseMeta(
            request_id=rid,
            provider_used="qa_extract_question_only",
            fallback_reason=reason_code,
            retrieval_count=retrieved["retrieval_count"],
            latency_ms=elapsed_ms,
        ),
        sources=fallback_sources or retrieved["sources"][:top_k],
    )
