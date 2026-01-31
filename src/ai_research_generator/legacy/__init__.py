"""Legacy code modules for backward compatibility.

This package contains the original monolithic modules that have been
refactored into the new package structure. They are kept for backward
compatibility and will be deprecated in future versions.
"""

from .academic_search import UnifiedAcademicSearch
from .llm_provider import LLMProvider, LLMConfig, ResearchLLMAssistant
from .research_engine import AIResearchEngine
from .main import main as legacy_main
from .ai_research_project_generator import AIResearchProjectGenerator
from .subject_analyzer import SubjectAnalyzer
from .validation_engine import ValidationEngine

__all__ = [
    "UnifiedAcademicSearch",
    "LLMProvider",
    "LLMConfig",
    "ResearchLLMAssistant",
    "AIResearchEngine",
    "legacy_main",
    "AIResearchProjectGenerator",
    "SubjectAnalyzer",
    "ValidationEngine",
]
