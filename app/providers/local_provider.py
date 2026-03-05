from __future__ import annotations

import httpx
from app.providers.base import BaseGeneratorProvider, ProviderError


class OllamaChatProvider(BaseGeneratorProvider):
    """
    Ollama chat provider via /api/chat
    """
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_sec: int = 60,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        num_ctx: int = 8192,
        max_tokens: int = 1024,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_sec = timeout_sec
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repeat_penalty = repeat_penalty
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens

    def generate(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "repeat_penalty": self.repeat_penalty,
                "num_ctx": self.num_ctx,
                "num_predict": self.max_tokens,
            },
        }

        try:
            with httpx.Client(timeout=self.timeout_sec) as client:
                r = client.post(f"{self.base_url}/api/chat", json=payload)
                r.raise_for_status()
                data = r.json()

            # Typical shape: {"message": {"role":"assistant","content":"..."}}
            msg = data.get("message") or {}
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()

            # Fallback: some versions return "response"
            resp = data.get("response")
            if isinstance(resp, str) and resp.strip():
                return resp.strip()

            raise ProviderError(f"Ollama returned unexpected response: {data}")

        except httpx.TimeoutException as exc:
            raise ProviderError(f"local_timeout: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"local_http_error: {exc}") from exc
        except Exception as exc:
            raise ProviderError(f"local_error: {exc}") from exc
