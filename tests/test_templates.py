"""Tests for research templates.

Tests the healthcare, technology, and investment research templates
including parameter validation, topic/question building, and template manager.
"""

import pytest

from src.ai_research_generator.templates import (
    HealthcareResearchTemplate,
    TechnologyResearchTemplate,
    InvestmentResearchTemplate,
    TemplateManager,
    get_template_manager,
)


class TestHealthcareTemplate:
    """Tests for HealthcareResearchTemplate."""

    @pytest.fixture
    def template(self):
        """Create healthcare template instance."""
        return HealthcareResearchTemplate()

    def test_template_metadata(self, template):
        """Test template has correct metadata."""
        assert template.template_id == "healthcare_research"
        assert template.name == "Healthcare Industry Research"
        assert template.category == "Healthcare"
        assert "healthcare" in template.tags
        assert "revenue-cycle" in template.tags

    def test_parameters_defined(self, template):
        """Test template has required parameters."""
        param_names = [p.name for p in template.parameters]
        
        # Check key parameters exist
        assert "healthcare_sector" in param_names
        assert "focus_area" in param_names
        assert "geographic_scope" in param_names
        assert "state" in param_names
        assert "payer_focus" in param_names
        assert "include_staffing_analysis" in param_names
        assert "executive_perspective" in param_names

    def test_build_topic_basic(self, template):
        """Test topic building with basic parameters."""
        params = {
            "healthcare_sector": "acute_care",
            "focus_area": "revenue_cycle",
            "geographic_scope": "state",
            "state": "Florida",
            "time_horizon": "3_years",
        }
        
        topic = template.build_topic(params)
        
        assert "revenue cycle management" in topic.lower()
        assert "acute care" in topic.lower()
        assert "florida" in topic.lower()
        assert "3 years" in topic.lower()

    def test_build_topic_national(self, template):
        """Test topic building with national scope."""
        params = {
            "healthcare_sector": "health_system",
            "focus_area": "clinical_operations",
            "geographic_scope": "national",
            "time_horizon": "5_years",
        }
        
        topic = template.build_topic(params)
        
        assert "clinical operations" in topic.lower()
        assert "united states" in topic.lower()

    def test_build_research_question(self, template):
        """Test research question building."""
        params = {
            "healthcare_sector": "acute_care",
            "focus_area": "revenue_cycle",
            "geographic_scope": "state",
            "state": "Florida",
            "executive_perspective": "vp_revenue_cycle",
        }
        
        question = template.build_research_question(params)
        
        assert "strategic adjustments" in question.lower()
        assert "florida" in question.lower()
        assert "vp revenue cycle" in question.lower()

    def test_build_additional_context(self, template):
        """Test additional context building."""
        params = {
            "healthcare_sector": "acute_care",
            "focus_area": "revenue_cycle",
            "geographic_scope": "state",
            "state": "Florida",
            "sub_focus_areas": ["charge_capture", "denial_management"],
            "payer_focus": ["medicare", "medicaid"],
            "include_staffing_analysis": True,
            "staffing_roles": ["coders", "cdi_specialists"],
            "include_technology_analysis": True,
            "specific_challenges": "High denial rates",
        }
        
        context = template.build_additional_context(params)
        
        assert "Healthcare Organization Profile" in context
        assert "Charge Capture" in context
        assert "Denial Management" in context
        assert "Medicare" in context
        assert "Staffing Analysis" in context
        assert "High denial rates" in context

    def test_get_search_keywords(self, template):
        """Test search keyword generation."""
        params = {
            "focus_area": "revenue_cycle",
            "state": "Florida",
            "sub_focus_areas": ["charge_capture"],
        }
        
        keywords = template.get_search_keywords(params)
        
        assert "healthcare" in keywords
        assert "revenue cycle management" in keywords
        assert "Florida healthcare" in keywords

    def test_to_research_request(self, template):
        """Test conversion to research request."""
        params = {
            "healthcare_sector": "acute_care",
            "focus_area": "revenue_cycle",
            "geographic_scope": "state",
            "state": "Florida",
        }
        
        request = template.to_research_request(params)
        
        assert "topic" in request
        assert "research_question" in request
        assert "discipline" in request
        assert request["discipline"] == "healthcare"
        assert request["search_papers"] is True

    def test_validate_params_valid(self, template):
        """Test parameter validation with valid params."""
        params = {
            "healthcare_sector": "acute_care",
            "focus_area": "revenue_cycle",
            "geographic_scope": "state",
        }
        
        is_valid, errors = template.validate_params(params)
        
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_params_invalid_sector(self, template):
        """Test parameter validation with invalid sector."""
        params = {
            "healthcare_sector": "invalid_sector",
            "focus_area": "revenue_cycle",
            "geographic_scope": "state",
        }
        
        is_valid, errors = template.validate_params(params)
        
        assert is_valid is False
        assert len(errors) > 0


class TestTechnologyTemplate:
    """Tests for TechnologyResearchTemplate."""

    @pytest.fixture
    def template(self):
        """Create technology template instance."""
        return TechnologyResearchTemplate()

    def test_template_metadata(self, template):
        """Test template has correct metadata."""
        assert template.template_id == "technology_research"
        assert template.name == "Technology Industry Research"
        assert template.category == "Technology"
        assert "technology" in template.tags
        assert "ai-ml" in template.tags

    def test_parameters_defined(self, template):
        """Test template has required parameters."""
        param_names = [p.name for p in template.parameters]
        
        assert "technology_domain" in param_names
        assert "research_focus" in param_names
        assert "target_industry" in param_names
        assert "organization_size" in param_names
        assert "include_vendor_analysis" in param_names

    def test_build_topic_ai(self, template):
        """Test topic building for AI domain."""
        params = {
            "technology_domain": "artificial_intelligence",
            "research_focus": "market_analysis",
            "target_industry": "healthcare",
            "time_horizon": "3_years",
        }
        
        topic = template.build_topic(params)
        
        assert "artificial intelligence" in topic.lower()
        assert "healthcare" in topic.lower()

    def test_build_topic_cloud(self, template):
        """Test topic building for cloud computing."""
        params = {
            "technology_domain": "cloud_computing",
            "research_focus": "vendor_comparison",
            "target_industry": "cross_industry",
            "time_horizon": "5_years",
        }
        
        topic = template.build_topic(params)
        
        assert "cloud computing" in topic.lower()
        assert "vendor comparison" in topic.lower()

    def test_build_research_question(self, template):
        """Test research question building."""
        params = {
            "technology_domain": "machine_learning",
            "research_focus": "adoption_strategy",
            "target_industry": "financial_services",
            "organization_size": "enterprise",
        }
        
        question = template.build_research_question(params)
        
        assert "machine learning" in question.lower()
        assert "enterprise" in question.lower()
        assert "financial services" in question.lower()

    def test_build_additional_context(self, template):
        """Test additional context building."""
        params = {
            "technology_domain": "artificial_intelligence",
            "research_focus": "market_analysis",
            "sub_domains": ["generative_ai", "natural_language_processing"],
            "include_market_sizing": True,
            "include_vendor_analysis": True,
            "specific_vendors": "OpenAI, Google, Microsoft",
            "adoption_barriers": ["skills_gap", "budget_constraints"],
        }
        
        context = template.build_additional_context(params)
        
        assert "Technology Research Profile" in context
        assert "Generative Ai" in context
        assert "Market Analysis Requirements" in context
        assert "OpenAI, Google, Microsoft" in context
        assert "Skills Gap" in context

    def test_get_search_keywords(self, template):
        """Test search keyword generation."""
        params = {
            "technology_domain": "artificial_intelligence",
            "sub_domains": ["generative_ai"],
            "target_industry": "healthcare",
        }
        
        keywords = template.get_search_keywords(params)
        
        assert "artificial intelligence" in keywords
        assert "AI" in keywords
        assert "healthcare technology" in keywords

    def test_to_research_request(self, template):
        """Test conversion to research request."""
        params = {
            "technology_domain": "cybersecurity",
            "research_focus": "technology_evaluation",
        }
        
        request = template.to_research_request(params)
        
        assert request["discipline"] == "technology"
        assert "cybersecurity" in request["topic"].lower()


class TestInvestmentTemplate:
    """Tests for InvestmentResearchTemplate."""

    @pytest.fixture
    def template(self):
        """Create investment template instance."""
        return InvestmentResearchTemplate()

    def test_template_metadata(self, template):
        """Test template has correct metadata."""
        assert template.template_id == "investment_research"
        assert template.name == "Investment Research"
        assert template.category == "Finance"

    def test_build_topic(self, template):
        """Test topic building."""
        params = {
            "investment_goal": "long_term_growth",
            "investment_horizon": 10,
            "asset_classes": ["stocks", "bonds", "etfs"],
        }
        
        topic = template.build_topic(params)
        
        assert "long-term" in topic.lower()
        assert "10-year" in topic.lower()

    def test_build_research_question(self, template):
        """Test research question building."""
        params = {
            "investment_goal": "retirement",
            "investment_horizon": 20,
            "risk_tolerance": "moderate",
        }
        
        question = template.build_research_question(params)
        
        assert "moderate" in question.lower()
        assert "20-year" in question.lower()


class TestTemplateManager:
    """Tests for TemplateManager."""

    @pytest.fixture
    def manager(self):
        """Create template manager instance."""
        return TemplateManager()

    def test_builtin_templates_registered(self, manager):
        """Test all built-in templates are registered."""
        templates = manager.list_templates()
        template_ids = [t["template_id"] for t in templates]
        
        assert "investment_research" in template_ids
        assert "healthcare_research" in template_ids
        assert "technology_research" in template_ids

    def test_get_template(self, manager):
        """Test getting template by ID."""
        template = manager.get("healthcare_research")
        
        assert template is not None
        assert template.template_id == "healthcare_research"

    def test_get_nonexistent_template(self, manager):
        """Test getting nonexistent template returns None."""
        template = manager.get("nonexistent_template")
        
        assert template is None

    def test_list_by_category(self, manager):
        """Test listing templates by category."""
        healthcare_templates = manager.list_by_category("Healthcare")
        
        assert len(healthcare_templates) >= 1
        assert healthcare_templates[0]["category"] == "Healthcare"

    def test_list_by_tag(self, manager):
        """Test listing templates by tag."""
        ai_templates = manager.list_by_tag("ai-ml")
        
        assert len(ai_templates) >= 1

    def test_get_categories(self, manager):
        """Test getting all categories."""
        categories = manager.get_categories()
        
        assert "Healthcare" in categories
        assert "Technology" in categories
        assert "Finance" in categories

    def test_get_tags(self, manager):
        """Test getting all tags."""
        tags = manager.get_tags()
        
        assert "healthcare" in tags
        assert "technology" in tags
        assert "investment" in tags

    def test_create_research_request(self, manager):
        """Test creating research request from template."""
        params = {
            "healthcare_sector": "acute_care",
            "focus_area": "revenue_cycle",
            "geographic_scope": "state",
            "state": "Texas",
        }
        
        request = manager.create_research_request("healthcare_research", params)
        
        assert "topic" in request
        assert "texas" in request["topic"].lower()

    def test_create_research_request_invalid_template(self, manager):
        """Test creating request with invalid template raises error."""
        with pytest.raises(ValueError) as exc_info:
            manager.create_research_request("invalid_template", {})
        
        assert "not found" in str(exc_info.value).lower()

    def test_validate_params(self, manager):
        """Test parameter validation through manager."""
        params = {
            "healthcare_sector": "acute_care",
            "focus_area": "revenue_cycle",
            "geographic_scope": "state",
        }
        
        is_valid, errors = manager.validate_params("healthcare_research", params)
        
        assert is_valid is True

    def test_get_defaults(self, manager):
        """Test getting default values."""
        defaults = manager.get_defaults("healthcare_research")
        
        assert "healthcare_sector" in defaults
        assert defaults["healthcare_sector"] == "acute_care"


class TestSingletonManager:
    """Tests for singleton template manager."""

    def test_get_template_manager_singleton(self):
        """Test get_template_manager returns same instance."""
        manager1 = get_template_manager()
        manager2 = get_template_manager()
        
        assert manager1 is manager2

    def test_singleton_has_templates(self):
        """Test singleton manager has templates registered."""
        manager = get_template_manager()
        templates = manager.list_templates()
        
        assert len(templates) >= 3


class TestParameterTypes:
    """Tests for parameter type validation."""

    def test_select_parameter_validation(self):
        """Test SELECT parameter validates options."""
        template = HealthcareResearchTemplate()
        sector_param = next(p for p in template.parameters if p.name == "healthcare_sector")
        
        # Valid value
        is_valid, error = sector_param.validate("acute_care")
        assert is_valid is True
        
        # Invalid value
        is_valid, error = sector_param.validate("invalid_value")
        assert is_valid is False

    def test_multi_select_parameter_validation(self):
        """Test MULTI_SELECT parameter validates options."""
        template = HealthcareResearchTemplate()
        payer_param = next(p for p in template.parameters if p.name == "payer_focus")
        
        # Valid values
        is_valid, error = payer_param.validate(["medicare", "medicaid"])
        assert is_valid is True
        
        # Invalid value in list
        is_valid, error = payer_param.validate(["medicare", "invalid_payer"])
        assert is_valid is False

    def test_integer_parameter_validation(self):
        """Test INTEGER parameter validates range."""
        template = HealthcareResearchTemplate()
        paper_param = next(p for p in template.parameters if p.name == "paper_limit")
        
        # Valid value
        is_valid, error = paper_param.validate(25)
        assert is_valid is True
        
        # Below minimum
        is_valid, error = paper_param.validate(5)
        assert is_valid is False

    def test_boolean_parameter_validation(self):
        """Test BOOLEAN parameter validates type."""
        template = HealthcareResearchTemplate()
        staffing_param = next(p for p in template.parameters if p.name == "include_staffing_analysis")
        
        # Valid value
        is_valid, error = staffing_param.validate(True)
        assert is_valid is True
        
        # Invalid type
        is_valid, error = staffing_param.validate("yes")
        assert is_valid is False
