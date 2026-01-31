"""
Research Tools Module

This module provides a collection of tools for AI-powered research workflows.
Tools follow LangGraph and PydanticAI patterns for integration with agent systems.
"""

from .research_tools import (
    WebSearchTool,
    AcademicSearchTool,
    KnowledgeSynthesisTool,
    FactVerificationTool,
    ResearchToolkit,
    search_web,
    search_academic_papers,
    synthesize_knowledge,
    verify_facts,
    generate_research_report,
)

__all__ = [
    "WebSearchTool",
    "AcademicSearchTool",
    "KnowledgeSynthesisTool",
    "FactVerificationTool",
    "ResearchToolkit",
    "search_web",
    "search_academic_papers",
    "synthesize_knowledge",
    "verify_facts",
    "generate_research_report",
]
