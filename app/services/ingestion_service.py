from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

from app.core.config import Settings
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import RAGRetriever
from app.storage.documents.loader import load_source_documents
from app.storage.embeddings.embedder import OllamaEmbedder
from app.storage.vectorstore.faiss_store import FaissStore
from app.utils.hashing import sha256_file
from app.services.text_normalizer import normalize_chars_fa


def _load_state(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_state(path: str, state: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def build_pipeline_from_existing_index(settings: Settings) -> Tuple[Optional[RAGPipeline], Dict[str, Any]]:
    source_hash = sha256_file(settings.static_source_path)
    state = _load_state(settings.index_state_path)

    # Require schema match (forces one-time rebuild when we change index design)
    if not state:
        return None, {"loaded": False, "reason": "missing_index_state", "source_hash": source_hash}

    if int(state.get("index_schema_version", -1)) != int(settings.index_schema_version):
        return None, {"loaded": False, "reason": "schema_version_mismatch", "state": state}

    if str(state.get("index_mode", "")) != str(settings.index_mode):
        return None, {"loaded": False, "reason": "index_mode_mismatch", "state": state}

    if str(state.get("embedding_model", "")) != str(settings.ollama_embed_model):
        return None, {"loaded": False, "reason": "embedding_model_mismatch", "state": state}

    if str(state.get("source_hash", "")) != str(source_hash):
        return None, {"loaded": False, "reason": "source_changed", "prev_hash": state.get("source_hash"), "source_hash": source_hash}

    store = FaissStore(
        index_path=settings.faiss_index_path,
        metadata_path=settings.faiss_metadata_path,
        embedding_dim=settings.embedding_dim,
    )
    if not store.load():
        return None, {"loaded": False, "reason": "faiss_missing_or_unreadable"}

    embedder = OllamaEmbedder(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embed_model,
        timeout_sec=settings.ollama_embed_timeout_sec,
    )
    retriever = RAGRetriever(store=store, embedder=embedder)
    pipeline = RAGPipeline(retriever=retriever, max_context_chars=settings.max_context_chars)

    return pipeline, {
        "loaded": True,
        "reason": "loaded_from_disk",
        "source_hash": source_hash,
        "index_state": state,
    }


def rebuild_index_and_pipeline(settings: Settings) -> Tuple[RAGPipeline, Dict[str, Any]]:
    docs, load_report = load_source_documents(settings.static_source_path)

    embedder = OllamaEmbedder(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embed_model,
        timeout_sec=settings.ollama_embed_timeout_sec,
    )

    store = FaissStore(
        index_path=settings.faiss_index_path,
        metadata_path=settings.faiss_metadata_path,
        embedding_dim=settings.embedding_dim,
    )

    items: List[Dict[str, Any]] = []
    embed_inputs: List[str] = []

    # qa_full: embed question + answer together.
    # Best for support FAQs where users describe their problem using words
    # that appear in the answer, not necessarily the question.
    if settings.index_mode in ("qa_full", "qa_question_only"):
        full_mode = settings.index_mode == "qa_full"
        for d in docs:
            q = (d.get("question") or "").strip()
            if not q:
                continue
            a = (d.get("answer") or "").strip()
            doc_id = d.get("doc_id")
            ri = (d.get("meta") or {}).get("record_index")

            items.append({
                "chunk_id": f"{doc_id}",
                "doc_id": doc_id,
                "record_index": ri,
                "question": q,
                "answer": a,
            })
            if full_mode and a:
                # Embed both sides so queries phrased like the answer also match
                embed_text = normalize_chars_fa(q + " " + a)
            else:
                embed_text = normalize_chars_fa(q)
            embed_inputs.append(embed_text)

    # Generic fallback for plain-text documents
    else:
        for d in docs:
            text = (d.get("text") or "").strip()
            if not text:
                continue
            items.append({
                "chunk_id": d.get("doc_id"),
                "doc_id": d.get("doc_id"),
                "question": text[:200],
                "answer": "",
            })
            embed_inputs.append(normalize_chars_fa(text))

    embeddings = embedder.embed_many(embed_inputs) if embed_inputs else []
    store.build_from_embeddings(items=items, embeddings=embeddings)
    store.save()

    source_hash = sha256_file(settings.static_source_path)
    state = {
        "index_schema_version": settings.index_schema_version,
        "index_mode": settings.index_mode,
        "source_path": settings.static_source_path,
        "source_hash": source_hash,
        "embedding_model": settings.ollama_embed_model,
        "items_indexed": len(items),
    }
    _save_state(settings.index_state_path, state)

    retriever = RAGRetriever(store=store, embedder=embedder)
    pipeline = RAGPipeline(retriever=retriever, max_context_chars=settings.max_context_chars)

    report = {
        **load_report,
        "loaded": False,
        "reason": "rebuilt_question_only",
        "items_indexed": len(items),
        "source_hash": source_hash,
        "index_state": state,
    }
    return pipeline, report
