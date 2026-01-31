"""
Custom exceptions for the research generation API

Provides specific exception types for different error scenarios.
"""

from typing import Optional


class ResearchGenerationError(Exception):
    """Base exception for research generation errors"""

    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message = message
        self.error_code = error_code or "RESEARCH_ERROR"
        super().__init__(self.message)


class LLMConnectionError(ResearchGenerationError):
    """Error connecting to or using LLM service"""

    def __init__(self, message: str):
        super().__init__(message, error_code="LLM_CONNECTION_ERROR")


class LLMGenerationError(ResearchGenerationError):
    """Error during LLM text generation"""

    def __init__(self, message: str):
        super().__init__(message, error_code="LLM_GENERATION_ERROR")


class AcademicSearchError(ResearchGenerationError):
    """Error during academic paper search"""

    def __init__(self, message: str, source: Optional[str] = None):
        self.source = source
        super().__init__(message, error_code="ACADEMIC_SEARCH_ERROR")


class ValidationError(ResearchGenerationError):
    """Error during project validation"""

    def __init__(self, message: str):
        super().__init__(message, error_code="VALIDATION_ERROR")


class ConfigurationError(ResearchGenerationError):
    """Error in application configuration"""

    def __init__(self, message: str):
        super().__init__(message, error_code="CONFIGURATION_ERROR")
