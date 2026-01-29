"""
Application configuration using Pydantic Settings

Loads configuration from environment variables with sensible defaults.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application
    app_name: str = "AI Research Project Generator"
    app_version: str = "2.0.0"
    debug: bool = False
    
    # API
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["*"]
    
    # LLM Configuration
    llm_provider: str = "ollama"
    llm_model: str = "llama3.1:8b"
    llm_base_url: str = "http://localhost:11434"
    llm_api_key: Optional[str] = None
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096
    llm_timeout: int = 120
    
    # Academic Search
    semantic_scholar_api_key: Optional[str] = None
    openalex_email: Optional[str] = None
    crossref_email: Optional[str] = None
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_period: int = 60  # seconds
    
    # Logging
    log_level: str = "INFO"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
