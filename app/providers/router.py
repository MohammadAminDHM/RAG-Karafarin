from __future__ import annotations

from app.providers.base import GenerationResult
from app.providers.local_provider import OllamaChatProvider
from app.providers.api_provider import OpenAICompatChatProvider
from app.services.busy_detector import BusyDetector, CircuitBreaker


class GeneratorRouter:
    def __init__(
        self,
        local: OllamaChatProvider,
        api: OpenAICompatChatProvider | None,
        busy: BusyDetector,
        circuit: CircuitBreaker,
        api_fallback_enabled: bool = True,
    ) -> None:
        self.local = local
        self.api = api
        self.busy = busy
        self.circuit = circuit
        self.api_fallback_enabled = bool(api_fallback_enabled)

    def generate(self, system: str, user: str) -> GenerationResult:
        # If circuit is open, skip local
        if self.circuit.is_open():
            if self.api_fallback_enabled and self.api is not None:
                ans = self.api.generate(system=system, user=user)
                return GenerationResult(
                    answer=ans,
                    provider_used="api",
                    fallback_reason="local_circuit_open",
                )
            return GenerationResult(
                answer="Local provider is temporarily unavailable (circuit open) and API fallback is disabled.",
                provider_used="none",
                fallback_reason="local_circuit_open_no_api",
            )

        # Busy detection (non-blocking)
        acq = self.busy.acquire_nowait()
        if not acq.acquired:
            if self.api_fallback_enabled and self.api is not None:
                ans = self.api.generate(system=system, user=user)
                return GenerationResult(
                    answer=ans,
                    provider_used="api",
                    fallback_reason=acq.reason,
                )
            return GenerationResult(
                answer="Local provider is busy and API fallback is disabled.",
                provider_used="none",
                fallback_reason="local_busy_no_api",
            )

        # Have local slot: try local then fallback on error/timeout
        try:
            ans = self.local.generate(system=system, user=user)
            self.circuit.record_success()
            return GenerationResult(answer=ans, provider_used="local")
        except Exception as exc:
            self.circuit.record_failure()
            if self.api_fallback_enabled and self.api is not None:
                ans = self.api.generate(system=system, user=user)
                return GenerationResult(
                    answer=ans,
                    provider_used="api",
                    fallback_reason=f"local_failed: {type(exc).__name__}",
                )
            return GenerationResult(
                answer=f"Local provider failed and API fallback is disabled. Error: {exc}",
                provider_used="none",
                fallback_reason="local_failed_no_api",
            )
        finally:
            self.busy.release()
