from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App metadata
    app_name: str = "rag-service"
    app_env: str = "dev"
    app_version: str = "0.1.0"

    # Server config
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # API config
    api_prefix: str = "/api/v1"

    # Logging
    log_level: str = "INFO"

    # Static source / chunking
    static_source_path: str = "./data/input/static_knowledge.txt"
    auto_ingest_on_startup: bool = True
    chunk_size: int = 900
    chunk_overlap: int = 150

    # Retrieval response shaping
    default_top_k: int = 5
    max_top_k: int = 10
    max_context_chars: int = 2400

    # Ollama embeddings
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_embed_timeout_sec: int = 30

    # FAISS persistence
    faiss_index_path: str = "./data/indexes/faiss.index"
    faiss_metadata_path: str = "./data/processed/faiss_chunks.json"

    # Index state (hash-based)
    index_state_path: str = "./data/processed/index_state.json"

    # Embedding dimension (0 => auto infer)
    embedding_dim: int = 0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> "Settings":
    return Settings()
