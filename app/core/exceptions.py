from __future__ import annotations
from typing import Any, Dict, Optional


class AppError(Exception):
    """
    A controlled exception that maps cleanly to an HTTP response.
    """
    def __init__(
        self,
        message: str,
        code: str = "app_error",
        status_code: int = 400,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
