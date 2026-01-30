"""Core application utilities and configuration"""

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    ResearchGenerationError,
    LLMConnectionError,
    AcademicSearchError,
    ValidationError,
)
from app.core.retry import (
    RetryConfig,
    retry_sync,
    retry_async,
    retry_operation,
    DEFAULT_RETRY_CONFIG,
    LLM_RETRY_CONFIG,
    ACADEMIC_SEARCH_RETRY_CONFIG,
)

__all__ = [
    "Settings",
    "get_settings",
    "ResearchGenerationError",
    "LLMConnectionError",
    "AcademicSearchError",
    "ValidationError",
    "RetryConfig",
    "retry_sync",
    "retry_async",
    "retry_operation",
    "DEFAULT_RETRY_CONFIG",
    "LLM_RETRY_CONFIG",
    "ACADEMIC_SEARCH_RETRY_CONFIG",
]
