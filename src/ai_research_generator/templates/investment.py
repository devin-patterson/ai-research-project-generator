"""Investment Research Template.

Specialized template for investment and financial research with
parameters optimized for analyzing investment instruments, market
conditions, and portfolio strategies.
"""

from dataclasses import dataclass, field
from typing import Any

from .base import ParameterType, ResearchTemplate, TemplateParameter


@dataclass
class InvestmentResearchTemplate(ResearchTemplate):
    """Template for comprehensive investment research.

    Provides specialized parameters for analyzing investment instruments,
    market conditions, risk factors, and portfolio strategies.
    """

    template_id: str = "investment_research"
    name: str = "Investment Research"
    description: str = (
        "Comprehensive investment analysis covering asset classes, market conditions, "
        "risk assessment, and portfolio strategies for informed investment decisions."
    )
    category: str = "Finance"
    version: str = "1.0.0"
    tags: list[str] = field(
        default_factory=lambda: [
            "investment",
            "finance",
            "portfolio",
            "market-analysis",
            "risk-assessment",
        ]
    )

    @property
    def parameters(self) -> list[TemplateParameter]:
        """Define investment-specific input parameters."""
        return [
            # Investment Focus Group
            TemplateParameter(
                name="investment_goal",
                display_name="Investment Goal",
                description="Primary investment objective",
                param_type=ParameterType.SELECT,
                required=True,
                default="long_term_growth",
                options=[
                    "long_term_growth",
                    "income_generation",
                    "capital_preservation",
                    "aggressive_growth",
                    "balanced",
                    "retirement",
                    "education_funding",
                    "wealth_transfer",
                ],
                group="Investment Focus",
            ),
            TemplateParameter(
                name="investment_horizon",
                display_name="Investment Horizon (Years)",
                description="Time horizon for the investment in years",
                param_type=ParameterType.INTEGER,
                required=True,
                default=10,
                min_value=1,
                max_value=50,
                help_text="Longer horizons allow for more aggressive strategies",
                group="Investment Focus",
            ),
            TemplateParameter(
                name="initial_investment",
                display_name="Initial Investment Amount ($)",
                description="Starting investment amount in USD",
                param_type=ParameterType.INTEGER,
                required=False,
                default=100000,
                min_value=1000,
                placeholder="100000",
                group="Investment Focus",
            ),
            TemplateParameter(
                name="monthly_contribution",
                display_name="Monthly Contribution ($)",
                description="Regular monthly investment amount",
                param_type=ParameterType.INTEGER,
                required=False,
                default=0,
                min_value=0,
                placeholder="1000",
                group="Investment Focus",
            ),
            # Risk Profile Group
            TemplateParameter(
                name="risk_tolerance",
                display_name="Risk Tolerance",
                description="Investor's risk tolerance level",
                param_type=ParameterType.SELECT,
                required=True,
                default="moderate",
                options=[
                    "conservative",
                    "moderately_conservative",
                    "moderate",
                    "moderately_aggressive",
                    "aggressive",
                ],
                help_text="Higher risk tolerance allows for more volatile investments",
                group="Risk Profile",
            ),
            TemplateParameter(
                name="max_drawdown_tolerance",
                display_name="Maximum Drawdown Tolerance (%)",
                description="Maximum acceptable portfolio decline",
                param_type=ParameterType.INTEGER,
                required=False,
                default=20,
                min_value=5,
                max_value=50,
                help_text="How much portfolio decline can you tolerate?",
                group="Risk Profile",
            ),
            TemplateParameter(
                name="volatility_preference",
                display_name="Volatility Preference",
                description="Preference for investment volatility",
                param_type=ParameterType.SELECT,
                required=False,
                default="moderate",
                options=["low", "moderate", "high"],
                group="Risk Profile",
            ),
            # Asset Classes Group
            TemplateParameter(
                name="asset_classes",
                display_name="Asset Classes to Consider",
                description="Investment asset classes to include in analysis",
                param_type=ParameterType.MULTI_SELECT,
                required=True,
                default=[
                    "stocks",
                    "bonds",
                    "etfs",
                    "real_estate",
                ],
                options=[
                    "stocks",
                    "bonds",
                    "etfs",
                    "mutual_funds",
                    "real_estate",
                    "reits",
                    "commodities",
                    "precious_metals",
                    "cryptocurrency",
                    "private_equity",
                    "hedge_funds",
                    "fixed_income",
                    "money_market",
                    "annuities",
                    "options",
                    "futures",
                ],
                help_text="Select all asset classes you want analyzed",
                group="Asset Classes",
            ),
            TemplateParameter(
                name="geographic_focus",
                display_name="Geographic Focus",
                description="Geographic regions for investment focus",
                param_type=ParameterType.MULTI_SELECT,
                required=False,
                default=["us", "international_developed"],
                options=[
                    "us",
                    "international_developed",
                    "emerging_markets",
                    "europe",
                    "asia_pacific",
                    "latin_america",
                    "global",
                ],
                group="Asset Classes",
            ),
            TemplateParameter(
                name="sector_preferences",
                display_name="Sector Preferences",
                description="Specific sectors of interest",
                param_type=ParameterType.MULTI_SELECT,
                required=False,
                default=[],
                options=[
                    "technology",
                    "healthcare",
                    "financials",
                    "consumer_discretionary",
                    "consumer_staples",
                    "industrials",
                    "energy",
                    "utilities",
                    "real_estate",
                    "materials",
                    "communication_services",
                ],
                group="Asset Classes",
            ),
            # Market Conditions Group
            TemplateParameter(
                name="market_outlook",
                display_name="Market Outlook Assumption",
                description="Expected market conditions",
                param_type=ParameterType.SELECT,
                required=False,
                default="neutral",
                options=[
                    "bullish",
                    "moderately_bullish",
                    "neutral",
                    "moderately_bearish",
                    "bearish",
                    "uncertain",
                ],
                group="Market Conditions",
            ),
            TemplateParameter(
                name="inflation_expectation",
                display_name="Inflation Expectation",
                description="Expected inflation environment",
                param_type=ParameterType.SELECT,
                required=False,
                default="moderate",
                options=[
                    "deflationary",
                    "low",
                    "moderate",
                    "high",
                    "hyperinflationary",
                ],
                help_text="Affects fixed income and real asset recommendations",
                group="Market Conditions",
            ),
            TemplateParameter(
                name="interest_rate_outlook",
                display_name="Interest Rate Outlook",
                description="Expected interest rate direction",
                param_type=ParameterType.SELECT,
                required=False,
                default="stable",
                options=[
                    "decreasing",
                    "stable",
                    "increasing",
                ],
                group="Market Conditions",
            ),
            TemplateParameter(
                name="economic_cycle_phase",
                display_name="Economic Cycle Phase",
                description="Current economic cycle position",
                param_type=ParameterType.SELECT,
                required=False,
                default="mid_cycle",
                options=[
                    "early_cycle",
                    "mid_cycle",
                    "late_cycle",
                    "recession",
                    "recovery",
                ],
                group="Market Conditions",
            ),
            # Investment Preferences Group
            TemplateParameter(
                name="dividend_preference",
                display_name="Dividend Preference",
                description="Preference for dividend-paying investments",
                param_type=ParameterType.SELECT,
                required=False,
                default="balanced",
                options=[
                    "no_preference",
                    "growth_focused",
                    "balanced",
                    "income_focused",
                    "high_yield",
                ],
                group="Investment Preferences",
            ),
            TemplateParameter(
                name="esg_preference",
                display_name="ESG/Sustainable Investing",
                description="Environmental, Social, Governance preference",
                param_type=ParameterType.SELECT,
                required=False,
                default="no_preference",
                options=[
                    "no_preference",
                    "esg_aware",
                    "esg_focused",
                    "impact_investing",
                    "exclusionary",
                ],
                group="Investment Preferences",
            ),
            TemplateParameter(
                name="tax_considerations",
                display_name="Tax Considerations",
                description="Tax situation affecting investment choices",
                param_type=ParameterType.SELECT,
                required=False,
                default="taxable",
                options=[
                    "taxable",
                    "tax_deferred",
                    "tax_free",
                    "mixed",
                ],
                help_text="Account type affects asset location strategy",
                group="Investment Preferences",
            ),
            TemplateParameter(
                name="liquidity_needs",
                display_name="Liquidity Needs",
                description="Need for accessible funds",
                param_type=ParameterType.SELECT,
                required=False,
                default="moderate",
                options=[
                    "high",
                    "moderate",
                    "low",
                ],
                group="Investment Preferences",
            ),
            # Research Depth Group
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
                name="include_historical_analysis",
                display_name="Include Historical Analysis",
                description="Include historical performance analysis",
                param_type=ParameterType.BOOLEAN,
                required=False,
                default=True,
                group="Research Settings",
            ),
            TemplateParameter(
                name="include_risk_metrics",
                display_name="Include Risk Metrics",
                description="Include detailed risk metrics (Sharpe, Sortino, etc.)",
                param_type=ParameterType.BOOLEAN,
                required=False,
                default=True,
                group="Research Settings",
            ),
            TemplateParameter(
                name="include_tax_strategies",
                display_name="Include Tax Strategies",
                description="Include tax optimization strategies",
                param_type=ParameterType.BOOLEAN,
                required=False,
                default=True,
                group="Research Settings",
            ),
            TemplateParameter(
                name="paper_limit",
                display_name="Academic Paper Limit",
                description="Maximum number of academic papers to analyze",
                param_type=ParameterType.INTEGER,
                required=False,
                default=30,
                min_value=10,
                max_value=100,
                group="Research Settings",
            ),
            # Additional Context
            TemplateParameter(
                name="specific_concerns",
                display_name="Specific Concerns or Questions",
                description="Any specific investment concerns or questions",
                param_type=ParameterType.TEXT,
                required=False,
                placeholder="e.g., Concerned about tech sector concentration, interested in dividend aristocrats",
                group="Additional Context",
            ),
            TemplateParameter(
                name="excluded_investments",
                display_name="Excluded Investments",
                description="Investments to exclude from recommendations",
                param_type=ParameterType.TEXT,
                required=False,
                placeholder="e.g., tobacco, weapons, fossil fuels",
                group="Additional Context",
            ),
        ]

    def build_topic(self, params: dict[str, Any]) -> str:
        """Build investment research topic from parameters."""
        goal_map = {
            "long_term_growth": "long-term wealth building",
            "income_generation": "income generation",
            "capital_preservation": "capital preservation",
            "aggressive_growth": "aggressive growth",
            "balanced": "balanced growth and income",
            "retirement": "retirement planning",
            "education_funding": "education funding",
            "wealth_transfer": "wealth transfer and estate planning",
        }

        goal = goal_map.get(params.get("investment_goal", "long_term_growth"), "investment")
        horizon = params.get("investment_horizon", 10)
        asset_classes = params.get("asset_classes", ["stocks", "bonds"])

        # Format asset classes nicely
        asset_str = ", ".join([a.replace("_", " ").title() for a in asset_classes[:4]])
        if len(asset_classes) > 4:
            asset_str += f", and {len(asset_classes) - 4} more"

        return (
            f"Optimal investment instruments for {goal} over a {horizon}-year horizon "
            f"focusing on {asset_str} based on current market conditions and forecasts"
        )

    def build_research_question(self, params: dict[str, Any]) -> str:
        """Build research question from parameters."""
        risk = params.get("risk_tolerance", "moderate").replace("_", " ")
        goal = params.get("investment_goal", "long_term_growth").replace("_", " ")
        horizon = params.get("investment_horizon", 10)

        return (
            f"What are the optimal investment instruments and allocation strategies for "
            f"a {risk} risk investor seeking {goal} over a {horizon}-year period, "
            f"considering current market conditions, economic forecasts, and risk-adjusted returns?"
        )

    def build_additional_context(self, params: dict[str, Any]) -> str:
        """Build comprehensive additional context from parameters."""
        context_parts = []

        # Investment Profile
        context_parts.append("## Investment Profile")
        context_parts.append(
            f"- **Goal**: {params.get('investment_goal', 'long_term_growth').replace('_', ' ').title()}"
        )
        context_parts.append(f"- **Time Horizon**: {params.get('investment_horizon', 10)} years")
        context_parts.append(
            f"- **Risk Tolerance**: {params.get('risk_tolerance', 'moderate').replace('_', ' ').title()}"
        )

        if params.get("initial_investment"):
            context_parts.append(f"- **Initial Investment**: ${params['initial_investment']:,}")
        if params.get("monthly_contribution"):
            context_parts.append(f"- **Monthly Contribution**: ${params['monthly_contribution']:,}")
        if params.get("max_drawdown_tolerance"):
            context_parts.append(
                f"- **Max Drawdown Tolerance**: {params['max_drawdown_tolerance']}%"
            )

        # Asset Classes
        context_parts.append("\n## Asset Classes to Analyze")
        asset_classes = params.get("asset_classes", [])
        for asset in asset_classes:
            context_parts.append(f"- {asset.replace('_', ' ').title()}")

        # Geographic Focus
        if params.get("geographic_focus"):
            context_parts.append("\n## Geographic Focus")
            for geo in params["geographic_focus"]:
                context_parts.append(f"- {geo.replace('_', ' ').title()}")

        # Sector Preferences
        if params.get("sector_preferences"):
            context_parts.append("\n## Sector Preferences")
            for sector in params["sector_preferences"]:
                context_parts.append(f"- {sector.replace('_', ' ').title()}")

        # Market Conditions
        context_parts.append("\n## Market Condition Assumptions")
        context_parts.append(
            f"- **Market Outlook**: {params.get('market_outlook', 'neutral').replace('_', ' ').title()}"
        )
        context_parts.append(
            f"- **Inflation Expectation**: {params.get('inflation_expectation', 'moderate').title()}"
        )
        context_parts.append(
            f"- **Interest Rate Outlook**: {params.get('interest_rate_outlook', 'stable').replace('_', ' ').title()}"
        )
        context_parts.append(
            f"- **Economic Cycle Phase**: {params.get('economic_cycle_phase', 'mid_cycle').replace('_', ' ').title()}"
        )

        # Investment Preferences
        context_parts.append("\n## Investment Preferences")
        context_parts.append(
            f"- **Dividend Preference**: {params.get('dividend_preference', 'balanced').replace('_', ' ').title()}"
        )
        context_parts.append(
            f"- **ESG Preference**: {params.get('esg_preference', 'no_preference').replace('_', ' ').title()}"
        )
        context_parts.append(
            f"- **Tax Situation**: {params.get('tax_considerations', 'taxable').replace('_', ' ').title()}"
        )
        context_parts.append(
            f"- **Liquidity Needs**: {params.get('liquidity_needs', 'moderate').title()}"
        )

        # Research Requirements
        context_parts.append("\n## Research Requirements")
        analysis_depth = params.get("analysis_depth", "comprehensive")
        context_parts.append(f"- **Analysis Depth**: {analysis_depth.title()}")

        if params.get("include_historical_analysis", True):
            context_parts.append("- Include historical performance analysis (5-10 year returns)")
        if params.get("include_risk_metrics", True):
            context_parts.append(
                "- Include risk metrics: Sharpe ratio, Sortino ratio, max drawdown, volatility"
            )
        if params.get("include_tax_strategies", True):
            context_parts.append(
                "- Include tax optimization strategies and asset location recommendations"
            )

        # Specific Requirements
        context_parts.append("\n## Specific Analysis Requirements")
        context_parts.append(
            "1. Provide specific investment instrument recommendations with rationale"
        )
        context_parts.append("2. Include suggested portfolio allocation percentages")
        context_parts.append("3. Discuss rebalancing strategies and frequency")
        context_parts.append("4. Address correlation between recommended assets")
        context_parts.append("5. Provide entry strategies (lump sum vs. dollar-cost averaging)")
        context_parts.append("6. Include risk mitigation strategies")
        context_parts.append("7. Discuss economic scenario analysis (bull/bear/base cases)")

        # Specific Concerns
        if params.get("specific_concerns"):
            context_parts.append("\n## Specific Concerns to Address")
            context_parts.append(params["specific_concerns"])

        # Exclusions
        if params.get("excluded_investments"):
            context_parts.append("\n## Excluded Investments")
            context_parts.append(f"Do NOT recommend: {params['excluded_investments']}")

        return "\n".join(context_parts)

    def get_search_keywords(self, params: dict[str, Any]) -> list[str]:
        """Generate optimized search keywords for investment research."""
        keywords = [
            "investment strategy",
            "portfolio allocation",
            "asset allocation",
            "risk-adjusted returns",
        ]

        # Add goal-specific keywords
        goal = params.get("investment_goal", "")
        if "growth" in goal:
            keywords.extend(["capital appreciation", "growth investing", "equity returns"])
        if "income" in goal:
            keywords.extend(["dividend investing", "income generation", "yield optimization"])
        if "preservation" in goal:
            keywords.extend(["capital preservation", "low volatility", "defensive investing"])

        # Add asset class keywords
        asset_classes = params.get("asset_classes", [])
        asset_keywords = {
            "stocks": ["equity investing", "stock selection"],
            "bonds": ["fixed income", "bond allocation"],
            "etfs": ["ETF investing", "passive investing"],
            "real_estate": ["real estate investing", "property investment"],
            "reits": ["REIT investing", "real estate investment trusts"],
            "commodities": ["commodity investing", "inflation hedge"],
            "cryptocurrency": ["digital assets", "crypto allocation"],
        }
        for asset in asset_classes:
            keywords.extend(asset_keywords.get(asset, []))

        # Add market condition keywords
        if params.get("inflation_expectation") in ["high", "hyperinflationary"]:
            keywords.extend(["inflation protection", "real assets", "TIPS"])
        if params.get("interest_rate_outlook") == "increasing":
            keywords.extend(["rising rates", "duration management"])

        return keywords

    def get_recommended_sources(self) -> list[str]:
        """Get recommended sources for investment research."""
        return ["openalex", "crossref", "semantic_scholar", "google_scholar"]

    def get_discipline(self) -> str:
        """Get primary discipline."""
        return "finance"

    def get_research_type(self) -> str:
        """Get recommended research type."""
        return "systematic_review"

    def get_academic_level(self) -> str:
        """Get recommended academic level."""
        return "professional"
