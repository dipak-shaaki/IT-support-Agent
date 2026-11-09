from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    app_name: str = "IT Support Agent"
    app_env: str = "development"
    cors_origins: str = "http://localhost:5173"
    groq_api_key: str = ""
    tavily_api_key: str = ""
    gemini_api_key: str = ""
    openai_api_key: str = ""
    pinecone_api_key: str = ""
    pinecone_index_name: str = "fde-rag-copilot"
    pinecone_namespace: str = "company-it-kb"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    groq_model: str = "openai/gpt-oss-120b"
    gemini_model: str = "gemini-3.8-flash"
    openai_model: str = "gpt-5.4-mini"
    top_k: int = 5
    redis_url: str = "redis://localhost:6379/0"
    conversation_ttl_seconds: int = Field(default=86400, gt=0)
    admin_api_key: str = "change-me"
    max_upload_bytes: int = Field(default=25 * 1024 * 1024, gt=0)
    auth_mode: Literal["none", "proxy"] = "none"
    auth_user_header: str = "X-Auth-Request-User"
    auth_roles_header: str = "X-Auth-Request-Roles"
    auth_admin_roles: str = ""
    rate_limit_per_minute: int = Field(default=30, gt=0)
    llm_timeout_seconds: int = Field(default=60, gt=0)
    ingest_mode: Literal["inline", "queue"] = "inline"
    ingest_job_ttl_seconds: int = Field(default=86400, gt=0)
    audit_db_path: str = str(BASE_DIR / "data" / "audit.db")
    bm25_index_path: str = str(BASE_DIR / "data" / "bm25_corpus.jsonl")
    upload_dir: str = str(BASE_DIR / "uploads")
    sample_kb_dir: str = str(BASE_DIR / "data" / "sample_kb")

    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), extra="ignore")

    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        if self.app_env == "production":
            if self.admin_api_key in {"", "change-me"}:
                raise ValueError(
                    "ADMIN_API_KEY must be set to a unique secret in production"
                )
            if "*" in self.cors_origins:
                raise ValueError("CORS_ORIGINS must not use a wildcard in production")
            if self.auth_mode == "none":
                raise ValueError(
                    "AUTH_MODE=proxy is required in production; terminate SSO at a "
                    "trusted reverse proxy and strip untrusted identity headers"
                )
        return self




@lru_cache
def get_settings() -> Settings:
    return Settings()
