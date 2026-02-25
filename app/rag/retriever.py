from typing import Dict, List

from app.storage.embeddings.embedder import OllamaEmbedder
from app.storage.vectorstore.base import VectorStoreProtocol


class RAGRetriever:
    def __init__(self, store: VectorStoreProtocol, embedder: OllamaEmbedder) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        qvec = self.embedder.embed_text(query)
        return self.store.search(query_vector=qvec, top_k=top_k)
