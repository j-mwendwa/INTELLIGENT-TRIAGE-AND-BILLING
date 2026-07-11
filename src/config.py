"""
src/config.py — Centralised settings and config loader.

Import everywhere as:
    from src.config import settings, cfg

Never read os.environ directly elsewhere.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"
DATA_DIR = BASE_DIR / "data"
MEMORY_DIR = DATA_DIR / "memory"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma"


# ── Pydantic Settings (from .env + environment) ──────────────────────────────
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # API Security
    allowed_api_keys: list[str] = Field(default=["dev-local-key"], alias="ALLOWED_API_KEYS")

    # Google Gemini
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")

    # Vector Store
    vector_backend: str = Field(default="chroma", alias="VECTOR_BACKEND")
    chroma_host: str = Field(default="localhost", alias="CHROMA_HOST")
    chroma_port: int = Field(default=8001, alias="CHROMA_PORT")
    chroma_collection: str = Field(default="knowledge_base", alias="CHROMA_COLLECTION")
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")

    # Memory encryption
    memory_encryption_key: str = Field(default="", alias="MEMORY_ENCRYPTION_KEY")

    # Observability
    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="triage-billing-agent", alias="LANGSMITH_PROJECT")

    # MCP
    mcp_enabled: bool = Field(default=True, alias="MCP_ENABLED")

    # CORS — comma-separated list of allowed origins for production.
    # Example: https://my-app.azurecontainerapps.io,https://my-custom-domain.com
    # In non-production environments ["*"] is used regardless of this value.
    cors_origins: list[str] = Field(default=[], alias="CORS_ORIGINS")

    # MCP optional
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")
    database_uri: str = Field(default="", alias="DATABASE_URI")

    @field_validator("allowed_api_keys", "cors_origins", mode="before")
    @classmethod
    def split_comma_list(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        return v

    @model_validator(mode="after")
    def production_safety_check(self) -> Settings:
        if self.app_env == "production":
            if "dev-local-key" in self.allowed_api_keys:
                raise ValueError(
                    "Production environment detected with default API key! "
                    "Set ALLOWED_API_KEYS to a secure value."
                )
            if not self.google_api_key:
                raise ValueError("GOOGLE_API_KEY must be set in production.")
        return self


# ── YAML Config loader ───────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_yaml_config() -> dict[str, Any]:
    """Load configs/config.yaml once and cache."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


class YamlConfig:
    """Dot-access wrapper around the YAML config dictionary."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        val = self._data.get(name)
        if isinstance(val, dict):
            return YamlConfig(val)
        return val

    def get(self, key: str, default: Any = None) -> Any:  # noqa: ANN401
        return self._data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __repr__(self) -> str:
        return f"YamlConfig({self._data!r})"


@lru_cache(maxsize=1)
def _get_settings() -> Settings:
    return Settings()


def _get_cfg() -> YamlConfig:
    # Always re-read for hot-reload; lru_cache cleared by make prompts-reload
    return YamlConfig(_load_yaml_config())


# ── Public exports ───────────────────────────────────────────────────────────
settings: Settings = _get_settings()
cfg: YamlConfig = _get_cfg()

# Ensure runtime directories exist
for _dir in (DATA_DIR, MEMORY_DIR, UPLOADS_DIR, CHROMA_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
