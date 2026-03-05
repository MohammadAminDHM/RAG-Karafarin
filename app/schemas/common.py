from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Human-readable error message")
    error_code: str = Field(..., description="Stable error code for clients")
    request_id: Optional[str] = Field(default=None, description="Request correlation id")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Optional debug details (safe)")


class StatusResponse(BaseModel):
    service: str
    env: str
    version: str
    api_prefix: str
    rag_ready: bool
    ingestion_report: Dict[str, Any]
    index_state: Dict[str, Any]
    config_safe: Dict[str, Any]
