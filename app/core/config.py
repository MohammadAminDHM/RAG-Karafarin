from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---- App ----
    app_name: str = "rag-service"
    app_env: str = "dev"
    app_version: str = "0.1.0"

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    api_prefix: str = "/api/v1"
    log_level: str = "INFO"

    # ---- Source / Index ----
    static_source_path: str = "./data/input/static_knowledge.txt"
    auto_ingest_on_startup: bool = True

    # chunk_size/overlap are only used in generic_chunks mode (plain-text docs).
    # For JSONL FAQ datasets (qa_full / qa_question_only) each record is its own unit
    # and these values have no effect on retrieval quality.
    chunk_size: int = 900
    chunk_overlap: int = 150

    default_top_k: int = 5
    max_top_k: int = 10
    max_context_chars: int = 4000  # context window passed to LLM if qa_mode=False

    # ---- Embedding (Ollama) ----
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_embed_timeout_sec: int = 60

    # ---- FAISS ----
    faiss_index_path: str = "./data/indexes/faiss.index"
    faiss_metadata_path: str = "./data/processed/faiss_chunks.json"
    index_state_path: str = "./data/processed/index_state.json"
    embedding_dim: int = 0

    # ---- Index schema/mode ----
    # Bump index_schema_version whenever index_mode or embedding_model changes
    # to force an automatic rebuild on next startup.
    index_schema_version: int = 3
    # qa_full   : embed question + answer — best recall for support FAQs
    # qa_question_only : embed question only — faster, fine for precise Q-Q matching
    # generic_chunks   : for plain-text documents
    index_mode: str = "qa_full"

    # ---- QA / rerank tuning ----
    # qa_mode=True: pure retrieval (no LLM call) — return best matching FAQ answer
    qa_mode: bool = True

    # How many candidates to pull from FAISS before reranking.
    # Higher = better recall, slightly slower.  100 is a good sweet spot for 1k-5k FAQ.
    qa_candidate_k: int = 100

    # Minimum cosine similarity (IndexFlatIP on L2-normalised vectors = cosine).
    # qwen3-embedding:8b on Persian: 0.25 reliably means same topic.
    # Lower → more recall, higher → more precision.
    qa_min_score: float = 0.25

    # Combined score gate (alpha*vector + beta*char + gamma*jaccard).
    # Should be at or slightly above qa_min_score.
    qa_min_combined: float = 0.25

    # Lexical gates are disabled — the embedding model is the primary trust signal.
    qa_min_match_char: float = 0.0
    qa_min_match_jaccard: float = 0.0
    qa_min_margin_to_second: float = 0.0

    rerank_enabled: bool = True
    # alpha: weight for FAISS cosine score  (primary — embedding model is strong)
    # beta : weight for char-level similarity (catches exact-phrase matches)
    # gamma: weight for Jaccard token overlap  (catches keyword overlap)
    # alpha+beta+gamma should sum to 1.0
    rerank_alpha: float = 0.70
    rerank_beta: float = 0.20
    rerank_gamma: float = 0.10

    # ---- LLM (Ollama chat) — used only when qa_mode=False ----
    # NOTE: these settings have NO effect on answer quality when qa_mode=True,
    # because the system returns the matched FAQ answer directly without any LLM call.
    ollama_chat_model: str = "gemma3:27b"
    ollama_chat_timeout_sec: int = 120
    ollama_chat_temperature: float = 0.3   # low = more factual / consistent
    ollama_chat_top_p: float = 0.85
    ollama_chat_top_k: int = 40
    ollama_chat_repeat_penalty: float = 1.05
    ollama_chat_num_ctx: int = 8192
    ollama_chat_max_tokens: int = 1024
    max_local_concurrent: int = 1

    # API fallback (OpenAI-compatible)
    api_fallback_enabled: bool = True
    api_base_url: str = "https://api.example.com"
    api_chat_path: str = "/v1/chat/completions"
    api_key: str = ""
    api_model: str = "gpt-4.1-mini"
    api_timeout_sec: int = 60
    api_max_retries: int = 2
    api_temperature: float = 0.2
    api_max_tokens: int = 1024
    api_top_p: float = 0.9

    # Circuit breaker
    local_fails_to_open_circuit: int = 3
    local_circuit_reset_sec: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> "Settings":
    return Settings()
