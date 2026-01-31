"""Base classes for research templates.

Defines the core template structure and parameter types for creating
specialized research templates.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ParameterType(str, Enum):
    """Types of template parameters."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    SELECT = "select"  # Single selection from options
    MULTI_SELECT = "multi_select"  # Multiple selections from options
    DATE = "date"
    DATE_RANGE = "date_range"
    TEXT = "text"  # Long-form text


@dataclass
class TemplateParameter:
    """Definition of a template input parameter."""

    name: str
    display_name: str
    description: str
    param_type: ParameterType
    required: bool = True
    default: Any = None
    options: list[str] = field(default_factory=list)  # For SELECT/MULTI_SELECT
    min_value: Optional[float] = None  # For numeric types
    max_value: Optional[float] = None  # For numeric types
    validation_regex: Optional[str] = None  # For string validation
    placeholder: Optional[str] = None
    help_text: Optional[str] = None
    group: Optional[str] = None  # For grouping related parameters

    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate a value against this parameter's constraints.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if value is None:
            if self.required:
                return False, f"{self.display_name} is required"
            return True, None

        if self.param_type == ParameterType.STRING:
            if not isinstance(value, str):
                return False, f"{self.display_name} must be a string"
            if self.validation_regex:
                import re

                if not re.match(self.validation_regex, value):
                    return False, f"{self.display_name} format is invalid"

        elif self.param_type == ParameterType.INTEGER:
            if not isinstance(value, int):
                return False, f"{self.display_name} must be an integer"
            if self.min_value is not None and value < self.min_value:
                return False, f"{self.display_name} must be at least {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"{self.display_name} must be at most {self.max_value}"

        elif self.param_type == ParameterType.FLOAT:
            if not isinstance(value, (int, float)):
                return False, f"{self.display_name} must be a number"
            if self.min_value is not None and value < self.min_value:
                return False, f"{self.display_name} must be at least {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"{self.display_name} must be at most {self.max_value}"

        elif self.param_type == ParameterType.BOOLEAN:
            if not isinstance(value, bool):
                return False, f"{self.display_name} must be true or false"

        elif self.param_type == ParameterType.SELECT:
            if value not in self.options:
                return False, f"{self.display_name} must be one of: {', '.join(self.options)}"

        elif self.param_type == ParameterType.MULTI_SELECT:
            if not isinstance(value, list):
                return False, f"{self.display_name} must be a list"
            invalid = [v for v in value if v not in self.options]
            if invalid:
                return False, f"Invalid options for {self.display_name}: {', '.join(invalid)}"

        return True, None


@dataclass
class ResearchTemplate(ABC):
    """Base class for research templates.

    Templates define specialized research configurations with
    domain-specific parameters, prompts, and search strategies.
    """

    template_id: str
    name: str
    description: str
    category: str
    version: str = "1.0.0"
    author: str = "AI Research Generator"
    tags: list[str] = field(default_factory=list)

    @property
    @abstractmethod
    def parameters(self) -> list[TemplateParameter]:
        """Define the input parameters for this template."""
        pass

    @abstractmethod
    def build_topic(self, params: dict[str, Any]) -> str:
        """Build the research topic from parameters."""
        pass

    @abstractmethod
    def build_research_question(self, params: dict[str, Any]) -> str:
        """Build the research question from parameters."""
        pass

    @abstractmethod
    def build_additional_context(self, params: dict[str, Any]) -> str:
        """Build additional context/instructions from parameters."""
        pass

    def get_search_keywords(self, params: dict[str, Any]) -> list[str]:
        """Generate optimized search keywords for academic databases."""
        return []

    def get_recommended_sources(self) -> list[str]:
        """Get recommended academic sources for this template."""
        return ["openalex", "crossref", "semantic_scholar"]

    def get_discipline(self) -> str:
        """Get the primary discipline for this template."""
        return "general"

    def get_research_type(self) -> str:
        """Get the recommended research type."""
        return "systematic_review"

    def get_academic_level(self) -> str:
        """Get the recommended academic level."""
        return "professional"

    def validate_params(self, params: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate all parameters.

        Returns:
            Tuple of (all_valid, list_of_errors)
        """
        errors = []
        for param in self.parameters:
            value = params.get(param.name, param.default)
            is_valid, error = param.validate(value)
            if not is_valid:
                errors.append(error)
        return len(errors) == 0, errors

    def get_defaults(self) -> dict[str, Any]:
        """Get default values for all parameters."""
        return {p.name: p.default for p in self.parameters if p.default is not None}

    def to_research_request(self, params: dict[str, Any]) -> dict[str, Any]:
        """Convert template parameters to a research request dict."""
        # Merge with defaults
        merged_params = {**self.get_defaults(), **params}

        # Validate
        is_valid, errors = self.validate_params(merged_params)
        if not is_valid:
            raise ValueError(f"Invalid parameters: {'; '.join(errors)}")

        return {
            "topic": self.build_topic(merged_params),
            "research_question": self.build_research_question(merged_params),
            "research_type": self.get_research_type(),
            "discipline": self.get_discipline(),
            "academic_level": self.get_academic_level(),
            "search_papers": True,
            "paper_limit": merged_params.get("paper_limit", 25),
            "use_llm": True,
            "additional_context": self.build_additional_context(merged_params),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize template metadata to dictionary."""
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "version": self.version,
            "author": self.author,
            "tags": self.tags,
            "discipline": self.get_discipline(),
            "research_type": self.get_research_type(),
            "parameters": [
                {
                    "name": p.name,
                    "display_name": p.display_name,
                    "description": p.description,
                    "type": p.param_type.value,
                    "required": p.required,
                    "default": p.default,
                    "options": p.options if p.options else None,
                    "min_value": p.min_value,
                    "max_value": p.max_value,
                    "placeholder": p.placeholder,
                    "help_text": p.help_text,
                    "group": p.group,
                }
                for p in self.parameters
            ],
        }
