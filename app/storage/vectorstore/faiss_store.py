from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import faiss
import numpy as np


class FaissStore:
    """
    FAISS store with cosine-like similarity using normalized vectors + IndexFlatIP.
    Persists:
      - index file
      - metadata JSON aligned with FAISS row ids
    """

    def __init__(self, index_path: str, metadata_path: str, embedding_dim: int = 0) -> None:
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.embedding_dim = int(embedding_dim or 0)

        self.index = None
        self.metadata: List[Dict] = []

    def _ensure_parent_dirs(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _to_float32_2d(vectors: Sequence[Sequence[float]]) -> np.ndarray:
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D vectors, got shape={arr.shape}")
        return np.ascontiguousarray(arr)

    @staticmethod
    def _to_float32_1d(vector: Sequence[float]) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D vector, got shape={arr.shape}")
        return np.ascontiguousarray(arr)

    @staticmethod
    def _l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return arr / norms

    @staticmethod
    def _l2_normalize_vector(arr: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(arr))
        if norm < 1e-12:
            return arr
        return arr / norm

    def _build_index(self, dim: int) -> None:
        self.index = faiss.IndexFlatIP(dim)

    def save(self) -> None:
        if self.index is None:
            raise ValueError("Cannot save FAISS index before building/loading it")
        self._ensure_parent_dirs()
        faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self) -> bool:
        if not self.index_path.exists() or not self.metadata_path.exists():
            return False
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.embedding_dim = int(self.index.d)
        return True

    def build_from_embeddings(self, chunks: List[Dict], embeddings: Sequence[Sequence[float]]) -> None:
        if not chunks:
            self.index = None
            self.metadata = []
            return

        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks/embeddings mismatch: {len(chunks)} != {len(embeddings)}")

        matrix = self._to_float32_2d(embeddings)
        inferred_dim = int(matrix.shape[1])

        if self.embedding_dim and self.embedding_dim != inferred_dim:
            raise ValueError(f"EMBEDDING_DIM mismatch: config={self.embedding_dim}, inferred={inferred_dim}")

        self.embedding_dim = inferred_dim
        self._build_index(self.embedding_dim)

        matrix = self._l2_normalize_rows(matrix)
        self.index.add(matrix)

        # Persist useful metadata (includes doc_id / record_index if present)
        keep_keys = ("chunk_id", "text", "start", "end", "doc_id", "doc_index", "record_index")
        self.metadata = []
        for c in chunks:
            meta = {k: c.get(k) for k in keep_keys if k in c}
            # Ensure minimum fields
            meta.setdefault("chunk_id", c.get("chunk_id"))
            meta.setdefault("text", c.get("text", ""))
            meta.setdefault("start", c.get("start"))
            meta.setdefault("end", c.get("end"))
            self.metadata.append(meta)

    def search(self, query_vector: Sequence[float], top_k: int = 5) -> List[Dict]:
        if self.index is None:
            loaded = self.load()
            if not loaded or self.index is None:
                return []

        q = self._to_float32_1d(query_vector)
        if q.shape[0] != self.embedding_dim:
            raise ValueError(f"Query dim mismatch: got={q.shape[0]}, expected={self.embedding_dim}")

        q = self._l2_normalize_vector(q).reshape(1, -1)
        k = max(1, min(int(top_k), len(self.metadata) if self.metadata else 1))

        scores, indices = self.index.search(q, k)
        out: List[Dict] = []

        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            out.append({**meta, "score": float(score)})

        return out
