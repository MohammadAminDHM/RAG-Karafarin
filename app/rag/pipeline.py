from typing import Dict, List

from app.rag.context_formatter import format_context_blocks
from app.rag.prompt_builder import build_rag_prompt


class RAGPipeline:
    """
    Phase 1 pipeline:
    retrieve (Ollama embeddings + FAISS) -> format context -> heuristic answer

    In later phases, heuristic answer generation will be replaced by local/API LLM.
    """

    def __init__(self, retriever, max_context_chars: int = 2400) -> None:
        self.retriever = retriever
        self.max_context_chars = max_context_chars

    def _heuristic_answer(self, query: str, results: List[Dict]) -> str:
        if not results:
            return "No relevant context was found in the indexed document."

        best = results[0]
        best_text = (best.get("text") or "").strip()

        preview = best_text[:700].strip()
        if len(best_text) > 700:
            preview += "..."

        return (
            "Based on the indexed document, the most relevant chunk is:\n"
            f"{preview}"
        )

    def ask(self, query: str, top_k: int = 5) -> Dict:
        results = self.retriever.retrieve(query=query, top_k=top_k)
        context = format_context_blocks(results, max_context_chars=self.max_context_chars)

        # Prepared for future generation phase
        _prompt = build_rag_prompt(query=query, context=context)

        answer = self._heuristic_answer(query=query, results=results)

        sources = [
            {
                "chunk_id": r.get("chunk_id"),
                "score": round(float(r.get("score", 0.0)), 4),
                "start": r.get("start"),
                "end": r.get("end"),
                "text_preview": (
                    ((r.get("text") or "")[:220] + "...")
                    if len(r.get("text") or "") > 220
                    else (r.get("text") or "")
                ),
            }
            for r in results
        ]

        return {
            "answer": answer,
            "sources": sources,
            "retrieval_count": len(results),
        }
