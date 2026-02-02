"""Technology Research Template.

Specialized template for technology industry research with
parameters optimized for emerging technologies, market analysis,
competitive dynamics, and technology adoption strategies.
"""

from dataclasses import dataclass, field
from typing import Any

from .base import ParameterType, ResearchTemplate, TemplateParameter


@dataclass
class TechnologyResearchTemplate(ResearchTemplate):
    """Template for comprehensive technology industry research.

    Provides specialized parameters for analyzing emerging technologies,
    market dynamics, competitive landscape, adoption patterns,
    and strategic technology decisions.
    """

    template_id: str = "technology_research"
    name: str = "Technology Industry Research"
    description: str = (
        "Comprehensive technology analysis covering emerging technologies, "
        "market dynamics, competitive landscape, adoption strategies, and "
        "implementation considerations for technology leaders and strategists."
    )
    category: str = "Technology"
    version: str = "1.0.0"
    tags: list[str] = field(
        default_factory=lambda: [
            "technology",
            "software",
            "ai-ml",
            "digital-transformation",
            "innovation",
        ]
    )

    @property
    def parameters(self) -> list[TemplateParameter]:
        """Define technology-specific input parameters."""
        return [
            # Technology Domain Group
            TemplateParameter(
                name="technology_domain",
                display_name="Technology Domain",
                description="Primary technology domain to research",
                param_type=ParameterType.SELECT,
                required=True,
                default="artificial_intelligence",
                options=[
                    "artificial_intelligence",
                    "machine_learning",
                    "cloud_computing",
                    "cybersecurity",
                    "data_analytics",
                    "blockchain",
                    "iot",
                    "edge_computing",
                    "quantum_computing",
                    "robotics_automation",
                    "ar_vr",
                    "5g_networking",
                    "devops_platform",
                    "low_code_no_code",
                ],
                group="Technology Focus",
            ),
            TemplateParameter(
                name="sub_domains",
                display_name="Sub-Domains",
                description="Specific technology sub-domains to analyze",
                param_type=ParameterType.MULTI_SELECT,
                required=False,
                default=[],
                options=[
                    # AI/ML
                    "generative_ai",
                    "natural_language_processing",
                    "computer_vision",
                    "predictive_analytics",
                    "mlops",
                    # Cloud
                    "iaas",
                    "paas",
                    "saas",
                    "serverless",
                    "containers_kubernetes",
                    "multi_cloud",
                    # Security
                    "zero_trust",
                    "identity_management",
                    "threat_detection",
                    "data_protection",
                    "devsecops",
                    # Data
                    "data_engineering",
                    "data_governance",
                    "real_time_analytics",
                    "data_lakehouse",
                ],
                group="Technology Focus",
            ),
            TemplateParameter(
                name="research_focus",
                display_name="Research Focus",
                description="Primary focus of the research",
                param_type=ParameterType.SELECT,
                required=True,
                default="market_analysis",
                options=[
                    "market_analysis",
                    "competitive_landscape",
                    "technology_evaluation",
                    "adoption_strategy",
                    "implementation_guide",
                    "vendor_comparison",
                    "roi_analysis",
                    "trend_forecast",
                ],
                group="Technology Focus",
            ),
            # Industry Application Group
            TemplateParameter(
                name="target_industry",
                display_name="Target Industry",
                description="Industry context for technology application",
                param_type=ParameterType.SELECT,
                required=False,
                default="cross_industry",
                options=[
                    "cross_industry",
                    "financial_services",
                    "healthcare",
                    "manufacturing",
                    "retail",
                    "telecommunications",
                    "energy_utilities",
                    "government",
                    "education",
                    "media_entertainment",
                    "transportation_logistics",
                ],
                group="Industry Context",
            ),
            TemplateParameter(
                name="organization_size",
                display_name="Organization Size",
                description="Target organization size for recommendations",
                param_type=ParameterType.SELECT,
                required=False,
                default="enterprise",
                options=[
                    "startup",
                    "smb",
                    "mid_market",
                    "enterprise",
                    "global_enterprise",
                ],
                group="Industry Context",
            ),
            # Market Analysis Group
            TemplateParameter(
                name="market_regions",
                display_name="Market Regions",
                description="Geographic markets to analyze",
                param_type=ParameterType.MULTI_SELECT,
                required=False,
                default=["north_america"],
                options=[
                    "north_america",
                    "europe",
                    "asia_pacific",
                    "latin_america",
                    "middle_east_africa",
                    "global",
                ],
                group="Market Analysis",
            ),
            TemplateParameter(
                name="include_market_sizing",
                display_name="Include Market Sizing",
                description="Include TAM/SAM/SOM analysis",
                param_type=ParameterType.BOOLEAN,
                required=False,
                default=True,
                group="Market Analysis",
            ),
            TemplateParameter(
                name="include_growth_projections",
                display_name="Include Growth Projections",
                description="Include market growth forecasts",
                param_type=ParameterType.BOOLEAN,
                required=False,
                default=True,
                group="Market Analysis",
            ),
            # Competitive Analysis Group
            TemplateParameter(
                name="include_vendor_analysis",
                display_name="Include Vendor Analysis",
                description="Include analysis of key vendors/players",
                param_type=ParameterType.BOOLEAN,
                required=False,
                default=True,
                group="Competitive Analysis",
            ),
            TemplateParameter(
                name="specific_vendors",
                display_name="Specific Vendors to Analyze",
                description="Specific vendors or companies to include",
                param_type=ParameterType.TEXT,
                required=False,
                placeholder="e.g., Microsoft, Google, AWS, Salesforce",
                group="Competitive Analysis",
            ),
            TemplateParameter(
                name="competitive_factors",
                display_name="Competitive Factors",
                description="Factors to evaluate in competitive analysis",
                param_type=ParameterType.MULTI_SELECT,
                required=False,
                default=["features", "pricing", "market_share"],
                options=[
                    "features",
                    "pricing",
                    "market_share",
                    "customer_satisfaction",
                    "innovation_pace",
                    "ecosystem_partnerships",
                    "enterprise_readiness",
                    "support_services",
                    "security_compliance",
                ],
                group="Competitive Analysis",
            ),
            # Adoption Analysis Group
            TemplateParameter(
                name="adoption_stage",
                display_name="Current Adoption Stage",
                description="Organization's current technology adoption stage",
                param_type=ParameterType.SELECT,
                required=False,
                default="evaluating",
                options=[
                    "exploring",
                    "evaluating",
                    "piloting",
                    "implementing",
                    "scaling",
                    "optimizing",
                ],
                group="Adoption Analysis",
            ),
            TemplateParameter(
                name="adoption_barriers",
                display_name="Adoption Barriers to Address",
                description="Key barriers to technology adoption",
                param_type=ParameterType.MULTI_SELECT,
                required=False,
                default=[],
                options=[
                    "budget_constraints",
                    "skills_gap",
                    "legacy_integration",
                    "security_concerns",
                    "regulatory_compliance",
                    "organizational_resistance",
                    "vendor_lock_in",
                    "roi_uncertainty",
                ],
                group="Adoption Analysis",
            ),
            # Implementation Group
            TemplateParameter(
                name="include_implementation_guide",
                display_name="Include Implementation Guide",
                description="Include implementation roadmap and best practices",
                param_type=ParameterType.BOOLEAN,
                required=False,
                default=True,
                group="Implementation",
            ),
            TemplateParameter(
                name="implementation_timeline",
                display_name="Implementation Timeline",
                description="Expected implementation timeline",
                param_type=ParameterType.SELECT,
                required=False,
                default="6_12_months",
                options=[
                    "0_3_months",
                    "3_6_months",
                    "6_12_months",
                    "12_24_months",
                    "24_plus_months",
                ],
                group="Implementation",
            ),
            TemplateParameter(
                name="budget_range",
                display_name="Budget Range",
                description="Approximate budget range for implementation",
                param_type=ParameterType.SELECT,
                required=False,
                default="medium",
                options=[
                    "minimal",
                    "low",
                    "medium",
                    "high",
                    "enterprise_scale",
                ],
                group="Implementation",
            ),
            # Research Settings Group
            TemplateParameter(
                name="time_horizon",
                display_name="Forecast Time Horizon",
                description="Time horizon for forecasts and projections",
                param_type=ParameterType.SELECT,
                required=False,
                default="3_years",
                options=[
                    "1_year",
                    "3_years",
                    "5_years",
                    "10_years",
                ],
                group="Research Settings",
            ),
            TemplateParameter(
                name="analysis_depth",
                display_name="Analysis Depth",
                description="Level of detail in the research",
                param_type=ParameterType.SELECT,
                required=False,
                default="comprehensive",
                options=[
                    "overview",
                    "standard",
                    "comprehensive",
                    "expert",
                ],
                group="Research Settings",
            ),
            TemplateParameter(
                name="paper_limit",
                display_name="Academic Paper Limit",
                description="Maximum number of academic papers to analyze",
                param_type=ParameterType.INTEGER,
                required=False,
                default=25,
                min_value=10,
                max_value=100,
                group="Research Settings",
            ),
            # Additional Context
            TemplateParameter(
                name="specific_questions",
                display_name="Specific Questions",
                description="Specific questions to address in the research",
                param_type=ParameterType.TEXT,
                required=False,
                placeholder="e.g., How does this compare to our current solution? What are the migration risks?",
                group="Additional Context",
            ),
            TemplateParameter(
                name="current_stack",
                display_name="Current Technology Stack",
                description="Current technology stack for integration analysis",
                param_type=ParameterType.TEXT,
                required=False,
                placeholder="e.g., AWS, Python, PostgreSQL, React",
                group="Additional Context",
            ),
        ]

    def build_topic(self, params: dict[str, Any]) -> str:
        """Build technology research topic from parameters."""
        domain_map = {
            "artificial_intelligence": "Artificial Intelligence",
            "machine_learning": "Machine Learning",
            "cloud_computing": "Cloud Computing",
            "cybersecurity": "Cybersecurity",
            "data_analytics": "Data Analytics",
            "blockchain": "Blockchain",
            "iot": "Internet of Things (IoT)",
            "edge_computing": "Edge Computing",
            "quantum_computing": "Quantum Computing",
            "robotics_automation": "Robotics and Automation",
            "ar_vr": "AR/VR Technologies",
            "5g_networking": "5G Networking",
            "devops_platform": "DevOps Platforms",
            "low_code_no_code": "Low-Code/No-Code Platforms",
        }

        focus_map = {
            "market_analysis": "market analysis and trends",
            "competitive_landscape": "competitive landscape analysis",
            "technology_evaluation": "technology evaluation and assessment",
            "adoption_strategy": "adoption strategy and roadmap",
            "implementation_guide": "implementation guide and best practices",
            "vendor_comparison": "vendor comparison and selection",
            "roi_analysis": "ROI analysis and business case",
            "trend_forecast": "trend forecasting and future outlook",
        }

        domain = domain_map.get(params.get("technology_domain", "artificial_intelligence"), "technology")
        focus = focus_map.get(params.get("research_focus", "market_analysis"), "analysis")

        # Add industry context
        industry = params.get("target_industry", "cross_industry")
        industry_context = ""
        if industry != "cross_industry":
            industry_context = f" for {industry.replace('_', ' ')} industry"

        # Add time horizon
        horizon = params.get("time_horizon", "3_years").replace("_", " ")

        return (
            f"Comprehensive {domain} {focus}{industry_context} "
            f"with {horizon} outlook and strategic recommendations"
        )

    def build_research_question(self, params: dict[str, Any]) -> str:
        """Build research question from parameters."""
        domain = params.get("technology_domain", "artificial_intelligence").replace("_", " ")
        org_size = params.get("organization_size", "enterprise").replace("_", " ")

        # Add industry context
        industry = params.get("target_industry", "cross_industry")
        industry_context = ""
        if industry != "cross_industry":
            industry_context = f" in the {industry.replace('_', ' ')} sector"

        return (
            f"What is the current state and future trajectory of {domain} technology, "
            f"and what strategic approach should {org_size} organizations{industry_context} "
            f"take for evaluation, adoption, and implementation to maximize business value?"
        )

    def build_additional_context(self, params: dict[str, Any]) -> str:
        """Build comprehensive additional context from parameters."""
        context_parts = []

        # Technology Profile
        context_parts.append("## Technology Research Profile")
        context_parts.append(
            f"- **Domain**: {params.get('technology_domain', 'artificial_intelligence').replace('_', ' ').title()}"
        )
        context_parts.append(
            f"- **Research Focus**: {params.get('research_focus', 'market_analysis').replace('_', ' ').title()}"
        )
        context_parts.append(
            f"- **Target Industry**: {params.get('target_industry', 'cross_industry').replace('_', ' ').title()}"
        )
        context_parts.append(
            f"- **Organization Size**: {params.get('organization_size', 'enterprise').replace('_', ' ').title()}"
        )

        # Sub-Domains
        if params.get("sub_domains"):
            context_parts.append("\n## Technology Sub-Domains")
            for sub in params["sub_domains"]:
                context_parts.append(f"- {sub.replace('_', ' ').title()}")

        # Market Analysis
        if params.get("include_market_sizing", True) or params.get("include_growth_projections", True):
            context_parts.append("\n## Market Analysis Requirements")
            if params.get("include_market_sizing", True):
                context_parts.append("- Include TAM/SAM/SOM market sizing")
            if params.get("include_growth_projections", True):
                context_parts.append("- Include market growth projections and CAGR")
            if params.get("market_regions"):
                regions = [r.replace("_", " ").title() for r in params["market_regions"]]
                context_parts.append(f"- Geographic focus: {', '.join(regions)}")

        # Competitive Analysis
        if params.get("include_vendor_analysis", True):
            context_parts.append("\n## Competitive Analysis Requirements")
            context_parts.append("- Analyze key vendors and market leaders")
            if params.get("specific_vendors"):
                context_parts.append(f"- Specific vendors to analyze: {params['specific_vendors']}")
            if params.get("competitive_factors"):
                factors = [f.replace("_", " ").title() for f in params["competitive_factors"]]
                context_parts.append(f"- Evaluation factors: {', '.join(factors)}")

        # Adoption Analysis
        context_parts.append("\n## Adoption Analysis")
        context_parts.append(
            f"- Current adoption stage: {params.get('adoption_stage', 'evaluating').replace('_', ' ').title()}"
        )
        if params.get("adoption_barriers"):
            context_parts.append("- Key barriers to address:")
            for barrier in params["adoption_barriers"]:
                context_parts.append(f"  - {barrier.replace('_', ' ').title()}")

        # Implementation Guide
        if params.get("include_implementation_guide", True):
            context_parts.append("\n## Implementation Requirements")
            timeline = params.get("implementation_timeline", "6_12_months").replace("_", "-")
            context_parts.append(f"- Target timeline: {timeline} months")
            context_parts.append(
                f"- Budget range: {params.get('budget_range', 'medium').replace('_', ' ').title()}"
            )
            context_parts.append("- Include implementation roadmap with phases")
            context_parts.append("- Include risk assessment and mitigation strategies")
            context_parts.append("- Include success metrics and KPIs")

        # Current Stack Integration
        if params.get("current_stack"):
            context_parts.append("\n## Integration Considerations")
            context_parts.append(f"Current technology stack: {params['current_stack']}")
            context_parts.append("- Analyze integration requirements and compatibility")
            context_parts.append("- Identify potential migration challenges")

        # Report Requirements
        context_parts.append("\n## Report Requirements")
        context_parts.append("1. Executive Summary with key findings and recommendations")
        context_parts.append("2. Technology landscape overview and maturity assessment")
        context_parts.append("3. Market analysis with sizing and growth projections")
        context_parts.append("4. Competitive analysis with vendor comparison matrix")
        context_parts.append("5. Adoption strategy with phased approach")
        context_parts.append("6. Implementation roadmap with timeline and milestones")
        context_parts.append("7. Risk assessment and mitigation strategies")
        context_parts.append("8. ROI analysis and business case")
        context_parts.append("9. Recommendations prioritized by impact and feasibility")

        # Specific Questions
        if params.get("specific_questions"):
            context_parts.append("\n## Specific Questions to Address")
            context_parts.append(params["specific_questions"])

        return "\n".join(context_parts)

    def get_search_keywords(self, params: dict[str, Any]) -> list[str]:
        """Generate optimized search keywords for technology research."""
        keywords = ["technology", "digital transformation"]

        # Add domain-specific keywords
        domain = params.get("technology_domain", "")
        domain_keywords = {
            "artificial_intelligence": ["artificial intelligence", "AI", "machine learning", "deep learning"],
            "machine_learning": ["machine learning", "ML", "neural networks", "model training"],
            "cloud_computing": ["cloud computing", "cloud infrastructure", "IaaS", "PaaS", "SaaS"],
            "cybersecurity": ["cybersecurity", "information security", "threat detection", "zero trust"],
            "data_analytics": ["data analytics", "business intelligence", "data science", "big data"],
            "blockchain": ["blockchain", "distributed ledger", "smart contracts", "web3"],
            "iot": ["internet of things", "IoT", "connected devices", "edge devices"],
        }
        keywords.extend(domain_keywords.get(domain, []))

        # Add sub-domain keywords
        for sub in params.get("sub_domains", []):
            keywords.append(sub.replace("_", " "))

        # Add industry keywords
        industry = params.get("target_industry", "")
        if industry and industry != "cross_industry":
            keywords.append(f"{industry.replace('_', ' ')} technology")

        return keywords

    def get_recommended_sources(self) -> list[str]:
        """Get recommended sources for technology research."""
        return ["openalex", "crossref", "semantic_scholar", "arxiv"]

    def get_discipline(self) -> str:
        """Get primary discipline."""
        return "technology"

    def get_research_type(self) -> str:
        """Get recommended research type."""
        return "systematic_review"

    def get_academic_level(self) -> str:
        """Get recommended academic level."""
        return "professional"
