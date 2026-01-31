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
    ToolConfig,
    search_web,
    search_academic_papers,
    synthesize_knowledge,
    verify_facts,
    generate_research_report,
)

from .google_scholar import (
    GoogleScholarTool,
    GoogleScholarConfig,
    GoogleScholarResult,
)

from .citation_manager import (
    CitationManager,
    CitationFormatter,
    Citation,
    CitationStyle,
    PublicationType,
)

__all__ = [
    # Research Tools
    "WebSearchTool",
    "AcademicSearchTool",
    "KnowledgeSynthesisTool",
    "FactVerificationTool",
    "ResearchToolkit",
    "ToolConfig",
    # Tool Functions
    "search_web",
    "search_academic_papers",
    "synthesize_knowledge",
    "verify_facts",
    "generate_research_report",
    # Google Scholar
    "GoogleScholarTool",
    "GoogleScholarConfig",
    "GoogleScholarResult",
    # Citation Management
    "CitationManager",
    "CitationFormatter",
    "Citation",
    "CitationStyle",
    "PublicationType",
]
