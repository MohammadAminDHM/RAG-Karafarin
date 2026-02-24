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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    # Cached settings to avoid repeated env parsing
    return Settings()
