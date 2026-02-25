from typing import Iterable, List, Sequence
import httpx


class OllamaEmbedder:
    """
    Ollama embeddings client.
    Tries /api/embed first (newer), then falls back to /api/embeddings (legacy).
    """

    def __init__(self, base_url: str, model: str, timeout_sec: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_sec = timeout_sec

    def _parse_embedding_response(self, data: dict) -> List[float]:
        """
        Parse both possible response formats:
        - {"embedding": [...]}
        - {"embeddings": [[...]]} or {"embeddings": [...]}
        """
        if "embedding" in data and isinstance(data["embedding"], list):
            return [float(x) for x in data["embedding"]]

        if "embeddings" in data and isinstance(data["embeddings"], list):
            embeddings = data["embeddings"]
            if not embeddings:
                raise ValueError("Ollama returned empty embeddings list")

            first = embeddings[0]
            # Case: embeddings is a single vector list[float]
            if isinstance(first, (int, float)):
                return [float(x) for x in embeddings]
            # Case: embeddings is list[list[float]]
            if isinstance(first, list):
                return [float(x) for x in first]

        raise ValueError(f"Unsupported Ollama embedding response format: {data}")

    def embed_text(self, text: str) -> List[float]:
        text = (text or "").strip()
        if not text:
            raise ValueError("Cannot embed empty text")

        with httpx.Client(timeout=self.timeout_sec) as client:
            # Try modern endpoint first
            try:
                resp = client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": text},
                )
                resp.raise_for_status()
                return self._parse_embedding_response(resp.json())
            except Exception:
                # Fallback to legacy endpoint
                resp = client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                )
                resp.raise_for_status()
                return self._parse_embedding_response(resp.json())

    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        cleaned: List[str] = [((t or "").strip()) for t in texts]
        cleaned = [t for t in cleaned if t]
        if not cleaned:
            return []

        # Try batch endpoint first
        with httpx.Client(timeout=self.timeout_sec) as client:
            try:
                resp = client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": cleaned},
                )
                resp.raise_for_status()
                data = resp.json()

                if "embeddings" in data and isinstance(data["embeddings"], list):
                    # Expected batch shape: list[list[float]]
                    if data["embeddings"] and isinstance(data["embeddings"][0], list):
                        return [
                            [float(x) for x in row]
                            for row in data["embeddings"]
                        ]

                # If format is unexpected, fall back to single-call loop
            except Exception:
                pass

        # Safe fallback: single-call embedding
        return [self.embed_text(t) for t in cleaned]

    @staticmethod
    def infer_dimension(vec: Sequence[float]) -> int:
        if not vec:
            raise ValueError("Cannot infer embedding dimension from empty vector")
        return len(vec)
