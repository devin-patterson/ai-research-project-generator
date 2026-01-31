"""
AI Research Project Generator

A comprehensive solution for generating robust research projects using AI.
Combines local LLM integration, academic search APIs, and rule-based project generation.
"""

__version__ = "2.0.0"
__author__ = "AI Research Assistant"

from .api.main import app
from .core.config import Settings
from .services.research_service import ResearchService

__all__ = ["app", "Settings", "ResearchService"]
