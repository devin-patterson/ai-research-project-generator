"""
Tests for research tools including CurrentEventsTool and EconomicDataTool.

This module tests the new data source tools:
1. CurrentEventsTool - Current events and news research
2. EconomicDataTool - Government economic data (FRED, BLS, BEA, Census, Treasury)
"""

import pytest
from datetime import datetime

from src.ai_research_generator.tools.current_events import (
    CurrentEventsTool,
    CurrentEventsConfig,
    CurrentEventsAnalysis,
    CurrentEvent,
    MarketCondition,
    format_current_events_markdown,
)
from src.ai_research_generator.tools.economic_data import (
    EconomicDataTool,
    EconomicDataConfig,
    EconomicDataReport,
    EconomicIndicator,
    EconomicSeries,
    EconomicDataPoint,
    FREDTool,
    BLSTool,
    TreasuryTool,
    format_economic_data_markdown,
    COMMON_INDICATORS,
)
from src.ai_research_generator.tools.research_tools import ToolConfig


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def tool_config():
    """Create a basic tool configuration."""
    return ToolConfig()


@pytest.fixture
def current_events_config():
    """Create a current events configuration."""
    return CurrentEventsConfig(
        max_search_results=10,
        search_queries_count=2,
        default_time_range_days=90,
    )


@pytest.fixture
def economic_data_config():
    """Create an economic data configuration without API keys."""
    return EconomicDataConfig(
        fred_api_key=None,
        bls_api_key=None,
        bea_api_key=None,
        census_api_key=None,
    )


@pytest.fixture
def sample_current_event():
    """Create a sample current event."""
    return CurrentEvent(
        title="Federal Reserve Raises Interest Rates",
        summary="The Fed raised rates by 25 basis points",
        date="2024-01-15",
        source="Reuters",
        url="https://reuters.com/article/123",
        relevance="Directly impacts investment returns",
        sentiment="neutral",
    )


@pytest.fixture
def sample_market_condition():
    """Create a sample market condition."""
    return MarketCondition(
        indicator="S&P 500",
        current_state="4,800",
        trend="rising",
        implications="Bullish market sentiment",
    )


@pytest.fixture
def sample_current_events_analysis(sample_current_event, sample_market_condition):
    """Create a sample current events analysis."""
    return CurrentEventsAnalysis(
        topic="Investment strategies",
        analysis_date="2024-01-20",
        executive_summary="Markets are showing positive momentum...",
        major_events=[sample_current_event],
        market_conditions=[sample_market_condition],
        emerging_trends=["AI-driven trading", "ESG investing"],
        current_risks=["Inflation concerns", "Geopolitical tensions"],
        current_opportunities=["Tech sector growth", "Emerging markets"],
        expert_perspectives=["Analysts expect continued growth"],
        actionable_insights=["Diversify portfolio", "Consider bonds"],
    )


@pytest.fixture
def sample_economic_indicator():
    """Create a sample economic indicator."""
    return EconomicIndicator(
        name="GDP Growth Rate",
        current_value="2.5%",
        previous_value="2.3%",
        change="0.2",
        change_percent="8.7%",
        date="2024-Q1",
        source="FRED",
        trend="rising",
    )


@pytest.fixture
def sample_economic_series():
    """Create a sample economic series."""
    return EconomicSeries(
        series_id="GDP",
        title="Gross Domestic Product",
        source="FRED",
        frequency="Quarterly",
        units="Billions of Dollars",
        observations=[
            EconomicDataPoint(
                date="2024-01-01",
                value="27000",
                series_id="GDP",
                source="FRED",
            ),
            EconomicDataPoint(
                date="2023-10-01",
                value="26500",
                series_id="GDP",
                source="FRED",
            ),
        ],
    )


# =============================================================================
# CurrentEventsConfig Tests
# =============================================================================


class TestCurrentEventsConfig:
    """Tests for CurrentEventsConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CurrentEventsConfig()

        assert config.max_search_results == 20
        assert config.search_queries_count == 4
        assert config.default_time_range_days == 180
        assert config.llm_timeout == 180.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CurrentEventsConfig(
            max_search_results=50,
            search_queries_count=6,
            default_time_range_days=365,
        )

        assert config.max_search_results == 50
        assert config.search_queries_count == 6
        assert config.default_time_range_days == 365


# =============================================================================
# CurrentEventsTool Tests
# =============================================================================


class TestCurrentEventsTool:
    """Tests for CurrentEventsTool."""

    def test_tool_properties(self, tool_config, current_events_config):
        """Test tool name and description."""
        tool = CurrentEventsTool(tool_config, current_events_config)

        assert tool.name == "current_events"
        assert "current events" in tool.description.lower()
        assert "news" in tool.description.lower()

    def test_tool_initialization(self, tool_config, current_events_config):
        """Test tool initialization with config."""
        tool = CurrentEventsTool(tool_config, current_events_config)

        assert tool.events_config == current_events_config
        assert tool.config == tool_config

    def test_tool_default_config(self, tool_config):
        """Test tool with default events config."""
        tool = CurrentEventsTool(tool_config)

        assert tool.events_config is not None
        assert isinstance(tool.events_config, CurrentEventsConfig)


# =============================================================================
# CurrentEventsAnalysis Tests
# =============================================================================


class TestCurrentEventsAnalysis:
    """Tests for CurrentEventsAnalysis model."""

    def test_analysis_creation(self, sample_current_events_analysis):
        """Test creating a current events analysis."""
        analysis = sample_current_events_analysis

        assert analysis.topic == "Investment strategies"
        assert len(analysis.major_events) == 1
        assert len(analysis.market_conditions) == 1
        assert len(analysis.emerging_trends) == 2
        assert len(analysis.current_risks) == 2

    def test_analysis_with_empty_lists(self):
        """Test analysis with empty optional lists."""
        analysis = CurrentEventsAnalysis(
            topic="Test",
            analysis_date="2024-01-01",
            executive_summary="Summary",
        )

        assert analysis.major_events == []
        assert analysis.market_conditions == []
        assert analysis.emerging_trends == []


# =============================================================================
# Current Events Markdown Formatter Tests
# =============================================================================


class TestCurrentEventsMarkdownFormatter:
    """Tests for current events markdown formatting."""

    def test_format_complete_analysis(self, sample_current_events_analysis):
        """Test formatting a complete analysis."""
        markdown = format_current_events_markdown(sample_current_events_analysis)

        assert "## Current Events Analysis" in markdown
        assert "Investment strategies" in markdown
        assert "Federal Reserve Raises Interest Rates" in markdown
        assert "S&P 500" in markdown
        assert "AI-driven trading" in markdown
        assert "Inflation concerns" in markdown

    def test_format_minimal_analysis(self):
        """Test formatting a minimal analysis."""
        analysis = CurrentEventsAnalysis(
            topic="Test Topic",
            analysis_date="2024-01-01",
            executive_summary="Brief summary",
        )

        markdown = format_current_events_markdown(analysis)

        assert "## Current Events Analysis" in markdown
        assert "Test Topic" in markdown
        assert "Brief summary" in markdown

    def test_format_includes_table_for_market_conditions(self, sample_current_events_analysis):
        """Test that market conditions are formatted as a table."""
        markdown = format_current_events_markdown(sample_current_events_analysis)

        assert "| Indicator |" in markdown
        assert "| S&P 500 |" in markdown


# =============================================================================
# EconomicDataConfig Tests
# =============================================================================


class TestEconomicDataConfig:
    """Tests for EconomicDataConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EconomicDataConfig()

        assert config.fred_base_url == "https://api.stlouisfed.org/fred"
        assert config.bls_base_url == "https://api.bls.gov/publicAPI/v2"
        assert (
            config.treasury_base_url
            == "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
        )
        assert config.http_timeout == 30.0

    def test_custom_api_keys(self):
        """Test configuration with custom API keys."""
        config = EconomicDataConfig(
            fred_api_key="test_fred_key",
            bls_api_key="test_bls_key",
        )

        assert config.fred_api_key == "test_fred_key"
        assert config.bls_api_key == "test_bls_key"


# =============================================================================
# EconomicDataTool Tests
# =============================================================================


class TestEconomicDataTool:
    """Tests for EconomicDataTool."""

    def test_tool_properties(self, tool_config, economic_data_config):
        """Test tool name and description."""
        tool = EconomicDataTool(tool_config, economic_data_config)

        assert tool.name == "economic_data"
        assert "FRED" in tool.description
        assert "BLS" in tool.description

    def test_tool_initialization(self, tool_config, economic_data_config):
        """Test tool initialization creates sub-tools."""
        tool = EconomicDataTool(tool_config, economic_data_config)

        assert tool.fred is not None
        assert tool.bls is not None
        assert tool.bea is not None
        assert tool.census is not None
        assert tool.treasury is not None

    def test_common_indicators_mapping(self):
        """Test that common indicators are properly mapped."""
        assert "gdp" in COMMON_INDICATORS
        assert "unemployment" in COMMON_INDICATORS
        assert "cpi" in COMMON_INDICATORS
        assert "fed_funds" in COMMON_INDICATORS

        # Check structure
        gdp_info = COMMON_INDICATORS["gdp"]
        assert "source" in gdp_info
        assert "series_id" in gdp_info
        assert "name" in gdp_info


# =============================================================================
# Individual Economic Data Tool Tests
# =============================================================================


class TestFREDTool:
    """Tests for FREDTool."""

    def test_tool_properties(self, tool_config, economic_data_config):
        """Test FRED tool properties."""
        tool = FREDTool(tool_config, economic_data_config)

        assert tool.name == "fred_data"
        assert "Federal Reserve" in tool.description

    @pytest.mark.asyncio
    async def test_execute_without_api_key(self, tool_config):
        """Test FRED tool returns empty without API key."""
        config = EconomicDataConfig(fred_api_key=None)
        tool = FREDTool(tool_config, config)

        result = await tool.execute(series_ids=["GDP"])

        assert result == []
        await tool.close()


class TestBLSTool:
    """Tests for BLSTool."""

    def test_tool_properties(self, tool_config, economic_data_config):
        """Test BLS tool properties."""
        tool = BLSTool(tool_config, economic_data_config)

        assert tool.name == "bls_data"
        assert "Bureau of Labor Statistics" in tool.description


class TestTreasuryTool:
    """Tests for TreasuryTool."""

    def test_tool_properties(self, tool_config, economic_data_config):
        """Test Treasury tool properties."""
        tool = TreasuryTool(tool_config, economic_data_config)

        assert tool.name == "treasury_data"
        assert "Treasury" in tool.description


# =============================================================================
# Economic Data Report Tests
# =============================================================================


class TestEconomicDataReport:
    """Tests for EconomicDataReport model."""

    def test_report_creation(self, sample_economic_indicator, sample_economic_series):
        """Test creating an economic data report."""
        report = EconomicDataReport(
            report_date="2024-01-20",
            topic="Economic Overview",
            indicators=[sample_economic_indicator],
            series_data=[sample_economic_series],
            summary="Economy is growing steadily",
            data_sources=["FRED", "BLS"],
        )

        assert report.report_date == "2024-01-20"
        assert len(report.indicators) == 1
        assert len(report.series_data) == 1
        assert len(report.data_sources) == 2

    def test_report_with_empty_data(self):
        """Test report with empty data."""
        report = EconomicDataReport(
            report_date="2024-01-20",
            topic="Test",
            summary="No data available",
        )

        assert report.indicators == []
        assert report.series_data == []
        assert report.data_sources == []


# =============================================================================
# Economic Data Markdown Formatter Tests
# =============================================================================


class TestEconomicDataMarkdownFormatter:
    """Tests for economic data markdown formatting."""

    def test_format_complete_report(self, sample_economic_indicator, sample_economic_series):
        """Test formatting a complete report."""
        report = EconomicDataReport(
            report_date="2024-01-20",
            topic="Economic Overview",
            indicators=[sample_economic_indicator],
            series_data=[sample_economic_series],
            summary="Economy is growing steadily",
            data_sources=["FRED", "BLS"],
        )

        markdown = format_economic_data_markdown(report)

        assert "## Economic Data Analysis" in markdown
        assert "FRED, BLS" in markdown
        assert "GDP Growth Rate" in markdown
        assert "2.5%" in markdown

    def test_format_includes_indicator_table(self, sample_economic_indicator):
        """Test that indicators are formatted as a table."""
        report = EconomicDataReport(
            report_date="2024-01-20",
            topic="Test",
            indicators=[sample_economic_indicator],
            summary="Test summary",
            data_sources=["FRED"],
        )

        markdown = format_economic_data_markdown(report)

        assert "| Indicator |" in markdown
        assert "| GDP Growth Rate |" in markdown


# =============================================================================
# Integration Tests
# =============================================================================


class TestToolsIntegration:
    """Integration tests for tools working together."""

    def test_current_events_tool_creates_valid_analysis(self):
        """Test that CurrentEventsTool output is valid for workflow."""
        analysis = CurrentEventsAnalysis(
            topic="Investment Research",
            analysis_date=datetime.now().strftime("%Y-%m-%d"),
            executive_summary="Market conditions are favorable",
            major_events=[
                CurrentEvent(
                    title="Market Rally",
                    summary="Stocks up 2%",
                    date="2024-01-15",
                    source="Bloomberg",
                    sentiment="positive",
                )
            ],
            market_conditions=[
                MarketCondition(
                    indicator="VIX",
                    current_state="15",
                    trend="falling",
                    implications="Low volatility",
                )
            ],
            emerging_trends=["AI stocks"],
            current_risks=["Rate hikes"],
            current_opportunities=["Tech sector"],
            expert_perspectives=["Bullish outlook"],
            actionable_insights=["Buy tech"],
        )

        # Verify can be converted to dict for workflow state
        analysis_dict = analysis.model_dump()

        assert "topic" in analysis_dict
        assert "major_events" in analysis_dict
        assert len(analysis_dict["major_events"]) == 1

    def test_economic_data_tool_creates_valid_report(self):
        """Test that EconomicDataTool output is valid for workflow."""
        report = EconomicDataReport(
            report_date=datetime.now().strftime("%Y-%m-%d"),
            topic="Economic Indicators",
            indicators=[
                EconomicIndicator(
                    name="Unemployment Rate",
                    current_value="3.7%",
                    date="2024-01",
                    source="BLS",
                    trend="stable",
                )
            ],
            series_data=[],
            summary="Economy stable",
            data_sources=["BLS"],
        )

        # Verify can be converted to dict for workflow state
        report_dict = {
            "report_date": report.report_date,
            "summary": report.summary,
            "data_sources": report.data_sources,
            "indicators": [ind.model_dump() for ind in report.indicators],
        }

        assert "report_date" in report_dict
        assert "indicators" in report_dict
        assert len(report_dict["indicators"]) == 1
