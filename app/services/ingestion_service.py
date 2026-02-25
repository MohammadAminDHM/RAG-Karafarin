from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import Settings
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import RAGRetriever
from app.storage.documents.cleaner import clean_text
from app.storage.documents.chunker import chunk_text
from app.storage.documents.loader import load_source_documents
from app.storage.embeddings.embedder import OllamaEmbedder
from app.storage.vectorstore.faiss_store import FaissStore
from app.utils.hashing import sha256_file


def _build_chunks_from_docs(
    docs: List[Dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict[str, Any]]:
    all_chunks: List[Dict[str, Any]] = []
    for doc_i, d in enumerate(docs):
        doc_id = d.get("doc_id", f"doc-{doc_i:06d}")
        meta = d.get("meta", {}) or {}
        cleaned = clean_text(d.get("text", "") or "")
        if not cleaned.strip():
            continue

        base_chunks = chunk_text(cleaned, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for ci, c in enumerate(base_chunks):
            all_chunks.append(
                {
                    "chunk_id": f"{doc_id}_chunk-{ci:04d}",
                    "text": c["text"],
                    "start": c["start"],
                    "end": c["end"],
                    "doc_id": doc_id,
                    "doc_index": doc_i,
                    **meta,
                }
            )
    return all_chunks


def _load_index_state(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_index_state(path: str, state: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def build_pipeline_from_existing_index(settings: Settings) -> Tuple[Optional[RAGPipeline], Dict[str, Any]]:
    """
    Fast path:
    - If FAISS index exists AND file hash hasn't changed => load only.
    - No chunking/embedding.
    """
    source_hash = sha256_file(settings.static_source_path)
    state = _load_index_state(settings.index_state_path)

    # Must have index files
    store = FaissStore(
        index_path=settings.faiss_index_path,
        metadata_path=settings.faiss_metadata_path,
        embedding_dim=settings.embedding_dim,
    )
    has_index = store.load()

    # If index missing => cannot load
    if not has_index:
        return None, {
            "loaded": False,
            "reason": "faiss_index_missing",
            "source_hash": source_hash,
        }

    # If hash mismatch => rebuild needed
    prev_hash = state.get("source_hash", "")
    if prev_hash and source_hash and prev_hash != source_hash:
        return None, {
            "loaded": False,
            "reason": "source_changed",
            "prev_hash": prev_hash,
            "source_hash": source_hash,
        }

    embedder = OllamaEmbedder(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embed_model,
        timeout_sec=settings.ollama_embed_timeout_sec,
    )
    retriever = RAGRetriever(store=store, embedder=embedder)
    pipeline = RAGPipeline(retriever=retriever, max_context_chars=settings.max_context_chars)

    report = {
        "loaded": True,
        "indexed": True,
        "reason": "loaded_from_disk",
        "faiss_index_path": settings.faiss_index_path,
        "faiss_metadata_path": settings.faiss_metadata_path,
        "index_state_path": settings.index_state_path,
        "source_hash": source_hash,
        "prev_hash": prev_hash,
    }
    return pipeline, report


def rebuild_index_and_pipeline(settings: Settings) -> Tuple[RAGPipeline, Dict[str, Any]]:
    """
    Slow path:
    - Load (txt/jsonl)
    - Chunk
    - Embed chunks with Ollama
    - Build + persist FAISS
    - Save state with source hash
    """
    embedder = OllamaEmbedder(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embed_model,
        timeout_sec=settings.ollama_embed_timeout_sec,
    )

    docs, load_report = load_source_documents(settings.static_source_path)
    chunks = _build_chunks_from_docs(docs, settings.chunk_size, settings.chunk_overlap)

    store = FaissStore(
        index_path=settings.faiss_index_path,
        metadata_path=settings.faiss_metadata_path,
        embedding_dim=settings.embedding_dim,
    )

    embedding_dim_actual = None
    if chunks:
        texts = [c["text"] for c in chunks]
        embeddings = embedder.embed_many(texts)
        if len(embeddings) != len(chunks):
            raise ValueError(f"Ollama embedding count mismatch: {len(embeddings)} != {len(chunks)}")
        embedding_dim_actual = len(embeddings[0]) if embeddings and embeddings[0] else None
        store.build_from_embeddings(chunks=chunks, embeddings=embeddings)
        store.save()
    else:
        store.build_from_embeddings(chunks=[], embeddings=[])

    source_hash = sha256_file(settings.static_source_path)
    _save_index_state(settings.index_state_path, {
        "source_path": settings.static_source_path,
        "source_hash": source_hash,
        "chunk_count": len(chunks),
        "embedding_model": settings.ollama_embed_model,
        "embedding_dim_actual": embedding_dim_actual,
    })

    retriever = RAGRetriever(store=store, embedder=embedder)
    pipeline = RAGPipeline(retriever=retriever, max_context_chars=settings.max_context_chars)

    report: Dict[str, Any] = {
        **load_report,
        "loaded": False,
        "indexed": bool(chunks),
        "reason": "rebuilt",
        "doc_count": len(docs),
        "chunk_count": len(chunks),
        "faiss_index_path": settings.faiss_index_path,
        "faiss_metadata_path": settings.faiss_metadata_path,
        "index_state_path": settings.index_state_path,
        "source_hash": source_hash,
        "embedding_model": settings.ollama_embed_model,
        "embedding_dim_config": settings.embedding_dim,
        "embedding_dim_actual": embedding_dim_actual,
    }
    return pipeline, report
