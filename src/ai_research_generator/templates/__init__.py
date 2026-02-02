"""Research Templates Module.

Provides pre-configured research templates for common use cases with
specialized input parameters for more valuable and robust research output."""

from .base import ParameterType, ResearchTemplate, TemplateParameter
from .investment import InvestmentResearchTemplate
from .healthcare import HealthcareResearchTemplate
from .technology import TechnologyResearchTemplate
from .manager import TemplateManager, get_template_manager

__all__ = [
    "ParameterType",
    "ResearchTemplate",
    "TemplateParameter",
    "InvestmentResearchTemplate",
    "HealthcareResearchTemplate",
    "TechnologyResearchTemplate",
    "TemplateManager",
    "get_template_manager",
]
