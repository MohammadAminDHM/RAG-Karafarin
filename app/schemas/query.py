from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    # User question
    query: str = Field(..., min_length=1, description="User query text")

    # Retrieval top-k
    top_k: int = Field(default=5, ge=1, le=20)

    # Optional client tracing
    client_request_id: Optional[str] = None


class QueryResponseMeta(BaseModel):
    provider_used: str = "phase1_rag_local_heuristic"
    fallback_reason: Optional[str] = None
    retrieval_count: int = 0
    latency_ms: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    meta: QueryResponseMeta
    sources: List[Dict[str, Any]] = []
