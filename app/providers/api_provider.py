from __future__ import annotations

import time
import httpx
from app.providers.base import BaseGeneratorProvider, ProviderError


class OpenAICompatChatProvider(BaseGeneratorProvider):
    """
    Generic OpenAI-compatible chat completions provider.
    Endpoint: {base_url}{chat_path}
    Payload: { model, messages, temperature }
    """
    def __init__(
        self,
        base_url: str,
        chat_path: str,
        api_key: str,
        model: str,
        timeout_sec: int = 60,
        max_retries: int = 2,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        top_p: float = 0.9,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.chat_path = chat_path if chat_path.startswith("/") else ("/" + chat_path)
        self.api_key = api_key
        self.model = model
        self.timeout_sec = timeout_sec
        self.max_retries = max(0, int(max_retries))
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

    def generate(self, system: str, user: str) -> str:
        if not self.api_key:
            raise ProviderError("api_key_missing")

        url = f"{self.base_url}{self.chat_path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }

        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_sec) as client:
                    r = client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()

                # OpenAI shape: choices[0].message.content
                choices = data.get("choices") or []
                if choices and isinstance(choices, list):
                    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
                    if msg and isinstance(msg.get("content"), str):
                        return msg["content"].strip()

                # Some providers return "output_text"
                if isinstance(data.get("output_text"), str):
                    return data["output_text"].strip()

                raise ProviderError(f"api_unexpected_response: {data}")

            except (httpx.TimeoutException, httpx.HTTPError, Exception) as exc:
                last_err = exc
                # simple backoff
                if attempt < self.max_retries:
                    time.sleep(0.4 * (attempt + 1))
                    continue
                break

        raise ProviderError(f"api_error: {last_err}") from last_err
