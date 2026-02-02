"""Healthcare Research Template.

Specialized template for healthcare industry research with
parameters optimized for revenue cycle, clinical operations,
regulatory compliance, and health IT analysis.
"""

from dataclasses import dataclass, field
from typing import Any

from .base import ParameterType, ResearchTemplate, TemplateParameter


@dataclass
class HealthcareResearchTemplate(ResearchTemplate):
    """Template for comprehensive healthcare industry research.

    Provides specialized parameters for analyzing healthcare operations,
    revenue cycle management, regulatory compliance, staffing models,
    and technology adoption.
    """

    template_id: str = "healthcare_research"
    name: str = "Healthcare Industry Research"
    description: str = (
        "Comprehensive healthcare analysis covering revenue cycle management, "
        "clinical operations, regulatory compliance, staffing models, and "
        "technology adoption for healthcare executives and administrators."
    )
    category: str = "Healthcare"
    version: str = "1.0.0"
    tags: list[str] = field(
        default_factory=lambda: [
            "healthcare",
            "revenue-cycle",
            "clinical-operations",
            "compliance",
            "health-it",
        ]
    )

    @property
    def parameters(self) -> list[TemplateParameter]:
        """Define healthcare-specific input parameters."""
        return [
            # Healthcare Sector Group
            TemplateParameter(
                name="healthcare_sector",
                display_name="Healthcare Sector",
                description="Primary healthcare sector focus",
                param_type=ParameterType.SELECT,
                required=True,
                default="acute_care",
                options=[
                    "acute_care",
                    "ambulatory_care",
                    "post_acute_care",
                    "behavioral_health",
                    "home_health",
                    "hospice",
                    "physician_practice",
                    "health_system",
                ],
                group="Healthcare Focus",
            ),
            TemplateParameter(
                name="focus_area",
                display_name="Primary Focus Area",
                description="Main area of research focus",
                param_type=ParameterType.SELECT,
                required=True,
                default="revenue_cycle",
                options=[
                    "revenue_cycle",
                    "clinical_operations",
                    "quality_safety",
                    "regulatory_compliance",
                    "health_it",
                    "workforce",
                    "patient_experience",
                    "supply_chain",
                    "strategic_planning",
                ],
                group="Healthcare Focus",
            ),
            TemplateParameter(
                name="sub_focus_areas",
                display_name="Sub-Focus Areas",
                description="Specific areas within the primary focus",
                param_type=ParameterType.MULTI_SELECT,
                required=False,
                default=[],
                options=[
                    # Revenue Cycle
                    "charge_capture",
                    "coding_accuracy",
                    "denial_management",
                    "patient_access",
                    "billing_collections",
                    "revenue_integrity",
                    "cdi_programs",
                    "payer_contracting",
                    # Clinical
                    "care_coordination",
                    "clinical_documentation",
                    "utilization_management",
                    "case_management",
                    # Compliance
                    "hipaa_compliance",
                    "cms_regulations",
                    "state_regulations",
                    "audit_response",
                    # Technology
                    "ehr_optimization",
                    "automation_rpa",
                    "ai_ml_applications",
                    "interoperability",
                ],
                group="Healthcare Focus",
            ),
            # Geographic Scope Group
            TemplateParameter(
                name="geographic_scope",
                display_name="Geographic Scope",
                description="Geographic scope of the research",
                param_type=ParameterType.SELECT,
                required=True,
                default="state",
                options=[
                    "national",
                    "regional",
                    "state",
                    "local",
                    "facility",
                ],
                group="Geographic Scope",
            ),
            TemplateParameter(
                name="state",
                display_name="State (if applicable)",
                description="Specific state for state-level analysis",
                param_type=ParameterType.STRING,
                required=False,
                default=None,
                placeholder="e.g., Florida, Texas, California",
                group="Geographic Scope",
            ),
            TemplateParameter(
                name="region",
                display_name="Region (if applicable)",
                description="Specific region for regional analysis",
                param_type=ParameterType.SELECT,
                required=False,
                default=None,
                options=[
                    "northeast",
                    "southeast",
                    "midwest",
                    "southwest",
                    "west",
                    "pacific",
                ],
                group="Geographic Scope",
            ),
            # Payer Mix Group
            TemplateParameter(
                name="payer_focus",
                display_name="Payer Focus",
                description="Primary payer types to analyze",
                param_type=ParameterType.MULTI_SELECT,
                required=False,
                default=["medicare", "medicaid", "commercial"],
                options=[
                    "medicare",
                    "medicare_advantage",
                    "medicaid",
                    "medicaid_managed_care",
                    "commercial",
                    "self_pay",
                    "workers_comp",
                    "tricare",
                ],
                group="Payer Analysis",
            ),
            TemplateParameter(
                name="reimbursement_models",
                display_name="Reimbursement Models",
                description="Payment models to consider",
                param_type=ParameterType.MULTI_SELECT,
                required=False,
                default=["fee_for_service", "value_based"],
                options=[
                    "fee_for_service",
                    "value_based",
                    "bundled_payments",
                    "capitation",
                    "drg_based",
                    "apc_based",
                ],
                group="Payer Analysis",
            ),
            # Regulatory Focus Group
            TemplateParameter(
                name="regulatory_areas",
                display_name="Regulatory Areas",
                description="Regulatory areas to analyze",
                param_type=ParameterType.MULTI_SELECT,
                required=False,
                default=[],
                options=[
                    "cms_conditions_of_participation",
                    "hipaa_privacy_security",
                    "stark_anti_kickback",
                    "price_transparency",
                    "no_surprises_act",
                    "information_blocking",
                    "state_licensure",
                    "joint_commission",
                    "cms_quality_programs",
                ],
                group="Regulatory Focus",
            ),
            # Staffing Analysis Group
            TemplateParameter(
                name="include_staffing_analysis",
                display_name="Include Staffing Analysis",
                description="Include workforce and staffing recommendations",
                param_type=ParameterType.BOOLEAN,
                required=False,
                default=True,
                group="Analysis Options",
            ),
            TemplateParameter(
                name="staffing_roles",
                display_name="Staffing Roles to Analyze",
                description="Specific roles to include in staffing analysis",
                param_type=ParameterType.MULTI_SELECT,
                required=False,
                default=[],
                options=[
                    "revenue_integrity_analysts",
                    "coders",
                    "cdi_specialists",
                    "billing_specialists",
                    "patient_access_reps",
                    "denial_management_specialists",
                    "compliance_officers",
                    "health_information_managers",
                    "revenue_cycle_directors",
                ],
                group="Analysis Options",
            ),
            # Technology Analysis Group
            TemplateParameter(
                name="include_technology_analysis",
                display_name="Include Technology Analysis",
                description="Include health IT and technology recommendations",
                param_type=ParameterType.BOOLEAN,
                required=False,
                default=True,
                group="Analysis Options",
            ),
            TemplateParameter(
                name="technology_areas",
                display_name="Technology Areas",
                description="Technology areas to analyze",
                param_type=ParameterType.MULTI_SELECT,
                required=False,
                default=[],
                options=[
                    "ehr_systems",
                    "revenue_cycle_systems",
                    "coding_software",
                    "cdi_technology",
                    "denial_management_tools",
                    "analytics_platforms",
                    "ai_automation",
                    "patient_portals",
                    "interoperability_solutions",
                ],
                group="Analysis Options",
            ),
            # Benchmarking Group
            TemplateParameter(
                name="include_benchmarks",
                display_name="Include Industry Benchmarks",
                description="Include comparison to industry benchmarks",
                param_type=ParameterType.BOOLEAN,
                required=False,
                default=True,
                group="Analysis Options",
            ),
            TemplateParameter(
                name="benchmark_sources",
                display_name="Benchmark Sources",
                description="Preferred benchmark data sources",
                param_type=ParameterType.MULTI_SELECT,
                required=False,
                default=["hfma", "aha"],
                options=[
                    "hfma",
                    "aha",
                    "mgma",
                    "cms_cost_reports",
                    "state_data",
                    "proprietary_surveys",
                ],
                group="Analysis Options",
            ),
            # Time Horizon Group
            TemplateParameter(
                name="time_horizon",
                display_name="Analysis Time Horizon",
                description="Time horizon for analysis and recommendations",
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
            # Executive Perspective Group
            TemplateParameter(
                name="executive_perspective",
                display_name="Executive Perspective",
                description="Target executive audience for recommendations",
                param_type=ParameterType.SELECT,
                required=False,
                default="vp_revenue_cycle",
                options=[
                    "ceo",
                    "cfo",
                    "coo",
                    "cmo",
                    "cio",
                    "cno",
                    "vp_revenue_cycle",
                    "vp_operations",
                    "vp_quality",
                    "compliance_officer",
                    "him_director",
                ],
                group="Output Settings",
            ),
            # Additional Context
            TemplateParameter(
                name="specific_challenges",
                display_name="Specific Challenges",
                description="Specific challenges or pain points to address",
                param_type=ParameterType.TEXT,
                required=False,
                placeholder="e.g., High denial rates for observation services, CDI program underperforming",
                group="Additional Context",
            ),
            TemplateParameter(
                name="organizational_context",
                display_name="Organizational Context",
                description="Relevant organizational context",
                param_type=ParameterType.TEXT,
                required=False,
                placeholder="e.g., 500-bed academic medical center, recent EHR implementation",
                group="Additional Context",
            ),
        ]

    def build_topic(self, params: dict[str, Any]) -> str:
        """Build healthcare research topic from parameters."""
        sector_map = {
            "acute_care": "acute care hospital",
            "ambulatory_care": "ambulatory care",
            "post_acute_care": "post-acute care",
            "behavioral_health": "behavioral health",
            "home_health": "home health",
            "hospice": "hospice",
            "physician_practice": "physician practice",
            "health_system": "integrated health system",
        }

        focus_map = {
            "revenue_cycle": "revenue cycle management",
            "clinical_operations": "clinical operations",
            "quality_safety": "quality and patient safety",
            "regulatory_compliance": "regulatory compliance",
            "health_it": "health information technology",
            "workforce": "workforce management",
            "patient_experience": "patient experience",
            "supply_chain": "supply chain management",
            "strategic_planning": "strategic planning",
        }

        sector = sector_map.get(params.get("healthcare_sector", "acute_care"), "healthcare")
        focus = focus_map.get(params.get("focus_area", "revenue_cycle"), "operations")

        # Build geographic context
        geo_context = ""
        if params.get("state"):
            geo_context = f" in {params['state']}"
        elif params.get("region"):
            geo_context = f" in the {params['region'].replace('_', ' ').title()} region"
        elif params.get("geographic_scope") == "national":
            geo_context = " across the United States"

        # Build time context
        horizon = params.get("time_horizon", "3_years").replace("_", " ")

        return (
            f"Strategic analysis of {focus} for {sector} organizations{geo_context} "
            f"with {horizon} outlook and actionable recommendations"
        )

    def build_research_question(self, params: dict[str, Any]) -> str:
        """Build research question from parameters."""
        focus = params.get("focus_area", "revenue_cycle").replace("_", " ")
        sector = params.get("healthcare_sector", "acute_care").replace("_", " ")
        perspective = params.get("executive_perspective", "vp_revenue_cycle").replace("_", " ").title()

        # Build geographic context
        geo_context = ""
        if params.get("state"):
            geo_context = f" in {params['state']}"
        elif params.get("geographic_scope") == "national":
            geo_context = " nationally"

        return (
            f"What strategic adjustments should {sector} organizations{geo_context} make to "
            f"{focus} methodology, staffing models, and technology adoption to optimize "
            f"performance and outcomes from a {perspective} perspective?"
        )

    def build_additional_context(self, params: dict[str, Any]) -> str:
        """Build comprehensive additional context from parameters."""
        context_parts = []

        # Healthcare Profile
        context_parts.append("## Healthcare Organization Profile")
        context_parts.append(
            f"- **Sector**: {params.get('healthcare_sector', 'acute_care').replace('_', ' ').title()}"
        )
        context_parts.append(
            f"- **Primary Focus**: {params.get('focus_area', 'revenue_cycle').replace('_', ' ').title()}"
        )
        context_parts.append(
            f"- **Geographic Scope**: {params.get('geographic_scope', 'state').title()}"
        )
        if params.get("state"):
            context_parts.append(f"- **State**: {params['state']}")
        context_parts.append(
            f"- **Executive Perspective**: {params.get('executive_perspective', 'vp_revenue_cycle').replace('_', ' ').title()}"
        )

        # Sub-Focus Areas
        if params.get("sub_focus_areas"):
            context_parts.append("\n## Specific Focus Areas")
            for area in params["sub_focus_areas"]:
                context_parts.append(f"- {area.replace('_', ' ').title()}")

        # Payer Analysis
        if params.get("payer_focus"):
            context_parts.append("\n## Payer Mix Analysis")
            for payer in params["payer_focus"]:
                context_parts.append(f"- {payer.replace('_', ' ').title()}")

        if params.get("reimbursement_models"):
            context_parts.append("\n## Reimbursement Models to Consider")
            for model in params["reimbursement_models"]:
                context_parts.append(f"- {model.replace('_', ' ').title()}")

        # Regulatory Focus
        if params.get("regulatory_areas"):
            context_parts.append("\n## Regulatory Areas to Address")
            for reg in params["regulatory_areas"]:
                context_parts.append(f"- {reg.replace('_', ' ').title()}")

        # Staffing Analysis
        if params.get("include_staffing_analysis", True):
            context_parts.append("\n## Staffing Analysis Requirements")
            context_parts.append("- Include FTE benchmarks and recommendations")
            context_parts.append("- Analyze skill requirements and certifications")
            context_parts.append("- Provide staffing ratios per net patient revenue")
            if params.get("staffing_roles"):
                context_parts.append("- Focus on these roles:")
                for role in params["staffing_roles"]:
                    context_parts.append(f"  - {role.replace('_', ' ').title()}")

        # Technology Analysis
        if params.get("include_technology_analysis", True):
            context_parts.append("\n## Technology Analysis Requirements")
            context_parts.append("- Assess current technology landscape")
            context_parts.append("- Identify automation and AI opportunities")
            context_parts.append("- Provide ROI analysis for technology investments")
            if params.get("technology_areas"):
                context_parts.append("- Focus on these technology areas:")
                for tech in params["technology_areas"]:
                    context_parts.append(f"  - {tech.replace('_', ' ').title()}")

        # Benchmarking
        if params.get("include_benchmarks", True):
            context_parts.append("\n## Benchmarking Requirements")
            context_parts.append("- Compare to industry benchmarks")
            context_parts.append("- Identify performance gaps")
            if params.get("benchmark_sources"):
                sources = [s.upper() for s in params["benchmark_sources"]]
                context_parts.append(f"- Reference sources: {', '.join(sources)}")

        # Time Horizon
        horizon = params.get("time_horizon", "3_years").replace("_", " ")
        context_parts.append(f"\n## Analysis Time Horizon: {horizon}")
        context_parts.append("- Provide current state assessment")
        context_parts.append("- Include near-term (1 year) tactical recommendations")
        context_parts.append(f"- Include strategic recommendations through {horizon}")

        # Report Requirements
        context_parts.append("\n## Report Requirements")
        context_parts.append("1. Executive Summary with key findings and recommendations")
        context_parts.append("2. Current state assessment with specific metrics")
        context_parts.append("3. Regulatory landscape and compliance considerations")
        context_parts.append("4. Operational recommendations with implementation priorities")
        context_parts.append("5. Staffing model recommendations with FTE benchmarks")
        context_parts.append("6. Technology roadmap with ROI projections")
        context_parts.append("7. Risk assessment and mitigation strategies")
        context_parts.append("8. Implementation timeline with milestones")

        # Specific Challenges
        if params.get("specific_challenges"):
            context_parts.append("\n## Specific Challenges to Address")
            context_parts.append(params["specific_challenges"])

        # Organizational Context
        if params.get("organizational_context"):
            context_parts.append("\n## Organizational Context")
            context_parts.append(params["organizational_context"])

        return "\n".join(context_parts)

    def get_search_keywords(self, params: dict[str, Any]) -> list[str]:
        """Generate optimized search keywords for healthcare research."""
        keywords = [
            "healthcare",
            "hospital",
            "health system",
        ]

        # Add focus-specific keywords
        focus = params.get("focus_area", "")
        focus_keywords = {
            "revenue_cycle": [
                "revenue cycle management",
                "charge capture",
                "denial management",
                "medical coding",
                "healthcare billing",
            ],
            "clinical_operations": [
                "clinical operations",
                "care coordination",
                "utilization management",
                "clinical documentation",
            ],
            "quality_safety": [
                "patient safety",
                "quality improvement",
                "clinical quality",
                "healthcare outcomes",
            ],
            "regulatory_compliance": [
                "healthcare compliance",
                "CMS regulations",
                "HIPAA",
                "healthcare audit",
            ],
            "health_it": [
                "health information technology",
                "EHR",
                "healthcare interoperability",
                "clinical informatics",
            ],
            "workforce": [
                "healthcare workforce",
                "staffing models",
                "healthcare human resources",
            ],
        }
        keywords.extend(focus_keywords.get(focus, []))

        # Add geographic keywords
        if params.get("state"):
            keywords.append(f"{params['state']} healthcare")

        # Add sub-focus keywords
        for sub in params.get("sub_focus_areas", []):
            keywords.append(sub.replace("_", " "))

        return keywords

    def get_recommended_sources(self) -> list[str]:
        """Get recommended sources for healthcare research."""
        return ["openalex", "crossref", "semantic_scholar", "pubmed"]

    def get_discipline(self) -> str:
        """Get primary discipline."""
        return "healthcare"

    def get_research_type(self) -> str:
        """Get recommended research type."""
        return "systematic_review"

    def get_academic_level(self) -> str:
        """Get recommended academic level."""
        return "professional"
