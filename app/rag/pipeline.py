from __future__ import annotations
from typing import Dict

class RAGPipeline:
    def __init__(self, retriever, max_context_chars: int = 2400) -> None:
        self.retriever = retriever
        self.max_context_chars = max_context_chars

    def retrieve(self, query: str, top_k: int = 5) -> Dict:
        results = self.retriever.retrieve(query=query, top_k=top_k)

        sources = []
        for r in results:
            q = (r.get("question") or "")[:160]
            a = (r.get("answer") or "")[:160]
            sources.append({
                "chunk_id": r.get("chunk_id"),
                "doc_id": r.get("doc_id"),
                "record_index": r.get("record_index"),
                "score": round(float(r.get("score", 0.0)), 4),
                "text_preview": f"Q: {q} | A: {a}".strip(),
            })

        return {
            "raw_results": results,
            "sources": sources,
            "retrieval_count": len(results),
        }
