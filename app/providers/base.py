from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class GenerationResult:
    answer: str
    provider_used: str
    fallback_reason: Optional[str] = None


class ProviderError(RuntimeError):
    pass


class BaseGeneratorProvider:
    def generate(self, system: str, user: str) -> str:
        raise NotImplementedError
