"""Template Manager for Research Templates.

Provides registration, discovery, and instantiation of research templates.
"""

from typing import Any, Optional

from loguru import logger

from .base import ResearchTemplate
from .investment import InvestmentResearchTemplate


class TemplateManager:
    """Manages research templates.

    Provides template registration, discovery, and instantiation.
    """

    def __init__(self):
        self._templates: dict[str, ResearchTemplate] = {}
        self._register_builtin_templates()

    def _register_builtin_templates(self):
        """Register all built-in templates."""
        builtin = [
            InvestmentResearchTemplate(),
        ]
        for template in builtin:
            self.register(template)
        logger.info(f"Registered {len(builtin)} built-in templates")

    def register(self, template: ResearchTemplate) -> None:
        """Register a template."""
        if template.template_id in self._templates:
            logger.warning(f"Overwriting existing template: {template.template_id}")
        self._templates[template.template_id] = template
        logger.debug(f"Registered template: {template.template_id}")

    def unregister(self, template_id: str) -> bool:
        """Unregister a template by ID."""
        if template_id in self._templates:
            del self._templates[template_id]
            return True
        return False

    def get(self, template_id: str) -> Optional[ResearchTemplate]:
        """Get a template by ID."""
        return self._templates.get(template_id)

    def list_templates(self) -> list[dict[str, Any]]:
        """List all available templates with metadata."""
        return [t.to_dict() for t in self._templates.values()]

    def list_by_category(self, category: str) -> list[dict[str, Any]]:
        """List templates filtered by category."""
        return [
            t.to_dict() for t in self._templates.values() if t.category.lower() == category.lower()
        ]

    def list_by_tag(self, tag: str) -> list[dict[str, Any]]:
        """List templates that have a specific tag."""
        return [
            t.to_dict()
            for t in self._templates.values()
            if tag.lower() in [x.lower() for x in t.tags]
        ]

    def get_categories(self) -> list[str]:
        """Get all unique categories."""
        return list(set(t.category for t in self._templates.values()))

    def get_tags(self) -> list[str]:
        """Get all unique tags."""
        tags = set()
        for t in self._templates.values():
            tags.update(t.tags)
        return sorted(tags)

    def create_research_request(self, template_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Create a research request from a template and parameters.

        Args:
            template_id: The template ID to use
            params: Parameter values for the template

        Returns:
            A dictionary suitable for creating a ResearchRequest

        Raises:
            ValueError: If template not found or parameters invalid
        """
        template = self.get(template_id)
        if not template:
            available = ", ".join(self._templates.keys())
            raise ValueError(f"Template '{template_id}' not found. Available: {available}")

        return template.to_research_request(params)

    def validate_params(self, template_id: str, params: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate parameters for a template.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        template = self.get(template_id)
        if not template:
            return False, [f"Template '{template_id}' not found"]
        return template.validate_params(params)

    def get_defaults(self, template_id: str) -> dict[str, Any]:
        """Get default parameter values for a template."""
        template = self.get(template_id)
        if not template:
            return {}
        return template.get_defaults()


# Singleton instance
_template_manager: Optional[TemplateManager] = None


def get_template_manager() -> TemplateManager:
    """Get the global template manager instance."""
    global _template_manager
    if _template_manager is None:
        _template_manager = TemplateManager()
    return _template_manager
