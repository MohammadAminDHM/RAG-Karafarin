from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query text")
    top_k: int = Field(default=5, ge=1, le=20)
    client_request_id: Optional[str] = None


class QueryResponseMeta(BaseModel):
    request_id: Optional[str] = None
    provider_used: str = "local"
    fallback_reason: Optional[str] = None
    retrieval_count: int = 0
    latency_ms: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    meta: QueryResponseMeta
    sources: List[Dict[str, Any]] = []
