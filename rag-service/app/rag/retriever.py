from typing import Dict, List

from app.storage.embeddings.embedder import OllamaEmbedder
from app.storage.vectorstore.base import VectorStoreProtocol
from app.services.text_normalizer import normalize_chars_fa


class RAGRetriever:
    def __init__(self, store: VectorStoreProtocol, embedder: OllamaEmbedder) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        # IMPORTANT: preprocess query only (no reindex required)
        q_norm = normalize_chars_fa(query)
        qvec = self.embedder.embed_text(q_norm)
        return self.store.search(query_vector=qvec, top_k=top_k)
