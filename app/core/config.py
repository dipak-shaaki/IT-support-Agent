from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    app_name: str = "IT Support Agent"
    app_env: str = "development"
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
    audit_db_path: str = str(BASE_DIR / "data" / "audit.db")
    bm25_index_path: str = str(BASE_DIR / "data" / "bm25_corpus.jsonl")
    upload_dir: str = str(BASE_DIR / "uploads")
    sample_kb_dir: str = str(BASE_DIR / "data" / "sample_kb")

    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), extra="ignore")




@lru_cache
def get_settings() -> Settings:
    return Settings()
