"""Research Templates Module.

Provides pre-configured research templates for common use cases with
specialized input parameters for more valuable and robust research output.
"""

from .base import ResearchTemplate, TemplateParameter, ParameterType
from .manager import TemplateManager, get_template_manager
from .investment import InvestmentResearchTemplate

__all__ = [
    "ResearchTemplate",
    "TemplateParameter",
    "ParameterType",
    "TemplateManager",
    "get_template_manager",
    "InvestmentResearchTemplate",
]
