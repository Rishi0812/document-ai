"""Application settings loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed configuration for the RAG application.

    Values are read from environment variables or a local `.env` file.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k: int = 5

    persist_dir: Path = Path("storage/chroma")
    uploads_dir: Path = Path("storage/uploads")
    collection_name: str = "knowledge_base"


def get_settings() -> Settings:
    """Return a fresh Settings instance."""
    return Settings()
