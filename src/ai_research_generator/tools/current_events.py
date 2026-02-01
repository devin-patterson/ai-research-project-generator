"""
Current Events Research Tool

This module provides a tool for researching current events, news, and market
conditions relevant to a research topic. It follows the same patterns as
other tools in the research_tools module.

Integrates with:
- Web search APIs (Serper, Tavily, DuckDuckGo)
- News APIs (optional)
- LLM for analysis and synthesis
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from .research_tools import BaseTool, ToolConfig, WebSearchTool


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CurrentEventsConfig:
    """Configuration for current events research."""

    # News API settings (optional)
    news_api_key: Optional[str] = field(default_factory=lambda: os.getenv("NEWS_API_KEY"))
    news_api_base_url: str = "https://newsapi.org/v2"

    # Search settings
    max_search_results: int = 20
    search_queries_count: int = 4

    # Time settings
    default_time_range_days: int = 180

    # LLM settings
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    llm_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.1:8b"))
    llm_timeout: float = 180.0


# =============================================================================
# Output Schemas
# =============================================================================


class CurrentEvent(BaseModel):
    """A single current event or news item."""

    title: str = Field(description="Title or headline of the event")
    summary: str = Field(description="Brief summary of the event")
    date: str = Field(description="Date of the event")
    source: str = Field(description="Source of the information")
    url: Optional[str] = Field(default=None, description="URL to the source")
    relevance: str = Field(default="", description="How this relates to the topic")
    sentiment: str = Field(default="neutral", description="positive/negative/neutral")


class MarketCondition(BaseModel):
    """Current market or economic condition."""

    indicator: str = Field(description="Name of the indicator")
    current_state: str = Field(description="Current state or value")
    trend: str = Field(description="Trend: rising/falling/stable")
    implications: str = Field(description="Implications for the topic")


class CurrentEventsAnalysis(BaseModel):
    """Structured output for current events research."""

    topic: str = Field(description="The research topic")
    analysis_date: str = Field(description="Date of analysis")
    executive_summary: str = Field(description="Executive summary")
    major_events: List[CurrentEvent] = Field(
        default_factory=list, description="Major current events"
    )
    market_conditions: List[MarketCondition] = Field(
        default_factory=list, description="Market conditions"
    )
    emerging_trends: List[str] = Field(default_factory=list, description="Emerging trends")
    current_risks: List[str] = Field(default_factory=list, description="Current risk factors")
    current_opportunities: List[str] = Field(
        default_factory=list, description="Current opportunities"
    )
    expert_perspectives: List[str] = Field(default_factory=list, description="Expert opinions")
    actionable_insights: List[str] = Field(default_factory=list, description="Actionable insights")


# =============================================================================
# Current Events Tool
# =============================================================================


class CurrentEventsTool(BaseTool):
    """Tool for researching current events and market conditions.

    This tool searches for recent news, analyzes market conditions,
    and synthesizes findings into actionable insights using LLM.
    """

    def __init__(
        self,
        config: Optional[ToolConfig] = None,
        events_config: Optional[CurrentEventsConfig] = None,
    ):
        super().__init__(config)
        self.events_config = events_config or CurrentEventsConfig()

    @property
    def name(self) -> str:
        return "current_events"

    @property
    def description(self) -> str:
        return (
            "Research current events, news, and market conditions relevant to a topic. "
            "Returns recent developments, trends, risks, and opportunities."
        )

    async def execute(
        self,
        topic: str,
        research_context: Optional[str] = None,
        include_market_conditions: bool = True,
        time_range_days: Optional[int] = None,
    ) -> CurrentEventsAnalysis:
        """
        Research current events for a topic.

        Args:
            topic: Research topic
            research_context: Additional context
            include_market_conditions: Include market analysis
            time_range_days: How far back to look (days), defaults to config value

        Returns:
            CurrentEventsAnalysis with findings
        """
        logger.info(f"Researching current events for: {topic[:50]}...")

        current_date = datetime.now().strftime("%Y-%m-%d")
        # time_range can be used for filtering results in future enhancements
        _ = time_range_days or self.events_config.default_time_range_days

        # Step 1: Search for recent news and events
        web_tool = WebSearchTool(self.config)
        current_year = datetime.now().year
        search_queries = [
            f"{topic} news {current_year}",
            f"{topic} recent developments {current_year}",
            f"{topic} market trends analysis",
            f"{topic} forecast outlook predictions",
        ]

        all_results = []
        results_per_query = self.events_config.max_search_results // len(search_queries)

        for query in search_queries[: self.events_config.search_queries_count]:
            try:
                results = await web_tool.execute(query=query, num_results=results_per_query)
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Search failed for '{query}': {e}")

        await web_tool.close()

        logger.info(f"Found {len(all_results)} search results for current events")

        # Step 2: Use LLM to analyze and structure the findings
        analysis = await self._analyze_with_llm(
            topic=topic,
            research_context=research_context or "",
            search_results=all_results,
            include_market_conditions=include_market_conditions,
            current_date=current_date,
        )

        return analysis

    async def _analyze_with_llm(
        self,
        topic: str,
        research_context: str,
        search_results: List[Any],
        include_market_conditions: bool,
        current_date: str,
    ) -> CurrentEventsAnalysis:
        """Analyze search results with LLM to extract current events."""

        # Format search results for prompt
        results_text = ""
        for i, r in enumerate(search_results[:15], 1):
            results_text += f"{i}. {r.title}\n   {r.snippet[:200]}\n   Source: {r.source}\n\n"

        market_section = ""
        if include_market_conditions:
            market_section = """
5. Market Conditions: Current market indicators and their trends
   - Include relevant economic indicators
   - Note current state and direction (rising/falling/stable)
   - Explain implications for the research topic"""

        prompt = f"""Analyze current events and developments for the following research topic.

Topic: {topic}
Research Context: {research_context}
Current Date: {current_date}

Recent Search Results:
{results_text}

Based on these results and your knowledge of recent events, provide a comprehensive current events analysis.

Include:
1. Executive Summary: 2-3 paragraph overview of the current landscape
2. Major Events: 3-5 significant recent events affecting this topic
3. Emerging Trends: Key trends developing in this area
4. Current Risks: Risk factors to be aware of
5. Current Opportunities: Opportunities in the current environment
{market_section}
6. Expert Perspectives: Recent expert opinions or forecasts
7. Actionable Insights: Practical recommendations based on current events

Format your response as JSON:
{{
    "executive_summary": "comprehensive summary...",
    "major_events": [
        {{"title": "Event Title", "summary": "Brief description", "date": "2024-XX", "source": "Source Name", "relevance": "How it relates", "sentiment": "positive/negative/neutral"}}
    ],
    "market_conditions": [
        {{"indicator": "Indicator Name", "current_state": "Current value/state", "trend": "rising/falling/stable", "implications": "What this means"}}
    ],
    "emerging_trends": ["trend1", "trend2"],
    "current_risks": ["risk1", "risk2"],
    "current_opportunities": ["opportunity1", "opportunity2"],
    "expert_perspectives": ["perspective1", "perspective2"],
    "actionable_insights": ["insight1", "insight2"]
}}"""

        try:
            response = await self.http_client.post(
                f"{self.events_config.llm_base_url}/api/generate",
                json={
                    "model": self.events_config.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                },
                timeout=self.events_config.llm_timeout,
            )
            response.raise_for_status()
            data = response.json()

            result = json.loads(data.get("response", "{}"))

            # Parse events
            events = []
            for e in result.get("major_events", []):
                if isinstance(e, dict):
                    events.append(
                        CurrentEvent(
                            title=e.get("title", "Unknown Event"),
                            summary=e.get("summary", ""),
                            date=e.get("date", current_date),
                            source=e.get("source", "Analysis"),
                            url=e.get("url"),
                            relevance=e.get("relevance", ""),
                            sentiment=e.get("sentiment", "neutral"),
                        )
                    )

            # Parse market conditions
            conditions = []
            for m in result.get("market_conditions", []):
                if isinstance(m, dict):
                    conditions.append(
                        MarketCondition(
                            indicator=m.get("indicator", "Unknown"),
                            current_state=m.get("current_state", "Unknown"),
                            trend=m.get("trend", "stable"),
                            implications=m.get("implications", ""),
                        )
                    )

            return CurrentEventsAnalysis(
                topic=topic,
                analysis_date=current_date,
                executive_summary=result.get("executive_summary", "Analysis unavailable"),
                major_events=events,
                market_conditions=conditions,
                emerging_trends=result.get("emerging_trends", []),
                current_risks=result.get("current_risks", []),
                current_opportunities=result.get("current_opportunities", []),
                expert_perspectives=result.get("expert_perspectives", []),
                actionable_insights=result.get("actionable_insights", []),
            )

        except Exception as e:
            logger.error(f"Current events analysis failed: {e}")
            return CurrentEventsAnalysis(
                topic=topic,
                analysis_date=current_date,
                executive_summary=f"Current events analysis for {topic}. Analysis encountered errors.",
                major_events=[],
                market_conditions=[],
                emerging_trends=[],
                current_risks=["Analysis incomplete - manual review recommended"],
                current_opportunities=[],
                expert_perspectives=[],
                actionable_insights=["Review current news sources manually"],
            )


# =============================================================================
# Markdown Formatter
# =============================================================================


def format_current_events_markdown(analysis: CurrentEventsAnalysis) -> str:
    """Format current events analysis as markdown."""
    lines = [
        "## Current Events Analysis",
        "",
        f"**Analysis Date:** {analysis.analysis_date}",
        f"**Topic:** {analysis.topic}",
        "",
        "### Executive Summary",
        "",
        analysis.executive_summary,
        "",
    ]

    # Major Events
    if analysis.major_events:
        lines.append("### Major Current Events")
        lines.append("")
        for i, event in enumerate(analysis.major_events, 1):
            lines.append(f"#### {i}. {event.title}")
            lines.append(f"- **Date:** {event.date}")
            lines.append(f"- **Source:** {event.source}")
            lines.append(f"- **Summary:** {event.summary}")
            if event.relevance:
                lines.append(f"- **Relevance:** {event.relevance}")
            lines.append(f"- **Sentiment:** {event.sentiment}")
            lines.append("")

    # Market Conditions
    if analysis.market_conditions:
        lines.append("### Current Market Conditions")
        lines.append("")
        lines.append("| Indicator | Current State | Trend | Implications |")
        lines.append("|-----------|---------------|-------|--------------|")
        for cond in analysis.market_conditions:
            lines.append(
                f"| {cond.indicator} | {cond.current_state} | {cond.trend} | {cond.implications} |"
            )
        lines.append("")

    # Emerging Trends
    if analysis.emerging_trends:
        lines.append("### Emerging Trends")
        lines.append("")
        for trend in analysis.emerging_trends:
            lines.append(f"- {trend}")
        lines.append("")

    # Risks
    if analysis.current_risks:
        lines.append("### Current Risk Factors")
        lines.append("")
        for risk in analysis.current_risks:
            lines.append(f"- ⚠️ {risk}")
        lines.append("")

    # Opportunities
    if analysis.current_opportunities:
        lines.append("### Current Opportunities")
        lines.append("")
        for opp in analysis.current_opportunities:
            lines.append(f"- ✅ {opp}")
        lines.append("")

    # Expert Perspectives
    if analysis.expert_perspectives:
        lines.append("### Expert Perspectives")
        lines.append("")
        for perspective in analysis.expert_perspectives:
            lines.append(f"- 💡 {perspective}")
        lines.append("")

    # Actionable Insights
    if analysis.actionable_insights:
        lines.append("### Actionable Insights")
        lines.append("")
        for i, insight in enumerate(analysis.actionable_insights, 1):
            lines.append(f"{i}. {insight}")
        lines.append("")

    return "\n".join(lines)
