"""
Economic Data Tools for Government Data Sources

This module provides tools for accessing economic data from various
U.S. government sources:
- FRED (Federal Reserve Economic Data)
- Bureau of Labor Statistics (BLS)
- Bureau of Economic Analysis (BEA)
- U.S. Census Bureau
- U.S. Treasury Fiscal Data
- Data.gov

Each tool follows the BaseTool pattern for integration with LangGraph workflows.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from .research_tools import BaseTool, ToolConfig


# =============================================================================
# Output Schemas
# =============================================================================


class EconomicDataPoint(BaseModel):
    """A single economic data observation."""

    date: str = Field(description="Date of the observation")
    value: str = Field(description="Value of the observation")
    series_id: str = Field(description="Series identifier")
    series_name: Optional[str] = Field(default=None, description="Human-readable name")
    source: str = Field(description="Data source (FRED, BLS, BEA, etc.)")
    unit: Optional[str] = Field(default=None, description="Unit of measurement")


class EconomicSeries(BaseModel):
    """An economic data series with metadata."""

    series_id: str = Field(description="Series identifier")
    title: str = Field(description="Series title")
    source: str = Field(description="Data source")
    frequency: Optional[str] = Field(default=None, description="Data frequency")
    units: Optional[str] = Field(default=None, description="Units")
    seasonal_adjustment: Optional[str] = Field(default=None, description="Seasonal adjustment")
    last_updated: Optional[str] = Field(default=None, description="Last update date")
    observations: List[EconomicDataPoint] = Field(
        default_factory=list, description="Data observations"
    )


class EconomicIndicator(BaseModel):
    """Summary of an economic indicator."""

    name: str = Field(description="Indicator name")
    current_value: str = Field(description="Most recent value")
    previous_value: Optional[str] = Field(default=None, description="Previous value")
    change: Optional[str] = Field(default=None, description="Change from previous")
    change_percent: Optional[str] = Field(default=None, description="Percent change")
    date: str = Field(description="Date of current value")
    source: str = Field(description="Data source")
    trend: str = Field(default="stable", description="Trend: rising/falling/stable")


class EconomicDataReport(BaseModel):
    """Comprehensive economic data report."""

    report_date: str = Field(description="Date of report generation")
    topic: str = Field(description="Research topic")
    indicators: List[EconomicIndicator] = Field(
        default_factory=list, description="Key economic indicators"
    )
    series_data: List[EconomicSeries] = Field(
        default_factory=list, description="Detailed series data"
    )
    summary: str = Field(description="Executive summary of economic conditions")
    data_sources: List[str] = Field(default_factory=list, description="Sources used")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class EconomicDataConfig:
    """Configuration for economic data APIs."""

    # FRED API
    fred_api_key: Optional[str] = field(default_factory=lambda: os.getenv("FRED_API_KEY"))
    fred_base_url: str = "https://api.stlouisfed.org/fred"

    # BLS API
    bls_api_key: Optional[str] = field(default_factory=lambda: os.getenv("BLS_API_KEY"))
    bls_base_url: str = "https://api.bls.gov/publicAPI/v2"

    # BEA API
    bea_api_key: Optional[str] = field(default_factory=lambda: os.getenv("BEA_API_KEY"))
    bea_base_url: str = "https://apps.bea.gov/api/data"

    # Census API
    census_api_key: Optional[str] = field(default_factory=lambda: os.getenv("CENSUS_API_KEY"))
    census_base_url: str = "https://api.census.gov/data"

    # Treasury Fiscal Data
    treasury_base_url: str = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"

    # HTTP settings
    http_timeout: float = 30.0


# =============================================================================
# FRED Tool
# =============================================================================


class FREDTool(BaseTool):
    """Tool for accessing Federal Reserve Economic Data (FRED)."""

    def __init__(
        self, config: Optional[ToolConfig] = None, econ_config: Optional[EconomicDataConfig] = None
    ):
        super().__init__(config)
        self.econ_config = econ_config or EconomicDataConfig()

    @property
    def name(self) -> str:
        return "fred_data"

    @property
    def description(self) -> str:
        return (
            "Access Federal Reserve Economic Data (FRED) for economic indicators "
            "like GDP, inflation, unemployment, interest rates, and more."
        )

    async def execute(
        self,
        series_ids: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
    ) -> List[EconomicSeries]:
        """
        Fetch data from FRED API.

        Args:
            series_ids: List of FRED series IDs (e.g., ['GDP', 'UNRATE', 'CPIAUCSL'])
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Maximum observations per series

        Returns:
            List of EconomicSeries with observations
        """
        logger.info(f"Fetching FRED data for series: {series_ids}")

        if not self.econ_config.fred_api_key:
            logger.warning("FRED API key not configured")
            return []

        results = []
        for series_id in series_ids:
            try:
                series = await self._fetch_series(series_id, start_date, end_date, limit)
                if series:
                    results.append(series)
            except Exception as e:
                logger.error(f"Error fetching FRED series {series_id}: {e}")

        return results

    async def _fetch_series(
        self,
        series_id: str,
        start_date: Optional[str],
        end_date: Optional[str],
        limit: int,
    ) -> Optional[EconomicSeries]:
        """Fetch a single FRED series."""
        params = {
            "series_id": series_id,
            "api_key": self.econ_config.fred_api_key,
            "file_type": "json",
            "limit": limit,
            "sort_order": "desc",
        }
        if start_date:
            params["observation_start"] = start_date
        if end_date:
            params["observation_end"] = end_date

        # Get series info
        info_response = await self.http_client.get(
            f"{self.econ_config.fred_base_url}/series",
            params={
                "series_id": series_id,
                "api_key": self.econ_config.fred_api_key,
                "file_type": "json",
            },
            timeout=self.econ_config.http_timeout,
        )
        info_response.raise_for_status()
        info_data = info_response.json()
        series_info = info_data.get("seriess", [{}])[0]

        # Get observations
        obs_response = await self.http_client.get(
            f"{self.econ_config.fred_base_url}/series/observations",
            params=params,
            timeout=self.econ_config.http_timeout,
        )
        obs_response.raise_for_status()
        obs_data = obs_response.json()

        observations = [
            EconomicDataPoint(
                date=obs["date"],
                value=obs["value"],
                series_id=series_id,
                series_name=series_info.get("title"),
                source="FRED",
            )
            for obs in obs_data.get("observations", [])
            if obs.get("value") != "."
        ]

        return EconomicSeries(
            series_id=series_id,
            title=series_info.get("title", series_id),
            source="FRED",
            frequency=series_info.get("frequency"),
            units=series_info.get("units"),
            seasonal_adjustment=series_info.get("seasonal_adjustment"),
            last_updated=series_info.get("last_updated"),
            observations=observations,
        )


# =============================================================================
# BLS Tool
# =============================================================================


class BLSTool(BaseTool):
    """Tool for accessing Bureau of Labor Statistics data."""

    def __init__(
        self, config: Optional[ToolConfig] = None, econ_config: Optional[EconomicDataConfig] = None
    ):
        super().__init__(config)
        self.econ_config = econ_config or EconomicDataConfig()

    @property
    def name(self) -> str:
        return "bls_data"

    @property
    def description(self) -> str:
        return (
            "Access Bureau of Labor Statistics data including employment, "
            "unemployment, CPI, wages, and productivity statistics."
        )

    async def execute(
        self,
        series_ids: List[str],
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> List[EconomicSeries]:
        """
        Fetch data from BLS API.

        Args:
            series_ids: List of BLS series IDs (e.g., ['CUUR0000SA0', 'LNS14000000'])
            start_year: Start year
            end_year: End year

        Returns:
            List of EconomicSeries with observations
        """
        logger.info(f"Fetching BLS data for series: {series_ids}")

        current_year = datetime.now().year
        start_year = start_year or current_year - 5
        end_year = end_year or current_year

        headers = {"Content-type": "application/json"}
        payload = {
            "seriesid": series_ids,
            "startyear": str(start_year),
            "endyear": str(end_year),
        }

        if self.econ_config.bls_api_key:
            payload["registrationkey"] = self.econ_config.bls_api_key

        try:
            response = await self.http_client.post(
                f"{self.econ_config.bls_base_url}/timeseries/data/",
                json=payload,
                headers=headers,
                timeout=self.econ_config.http_timeout,
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for series in data.get("Results", {}).get("series", []):
                series_id = series.get("seriesID", "")
                observations = []

                for item in series.get("data", []):
                    period = item.get("period", "")
                    if period.startswith("M"):
                        month = period[1:]
                        date = f"{item.get('year')}-{month}-01"
                        observations.append(
                            EconomicDataPoint(
                                date=date,
                                value=item.get("value", ""),
                                series_id=series_id,
                                source="BLS",
                            )
                        )

                results.append(
                    EconomicSeries(
                        series_id=series_id,
                        title=series_id,
                        source="BLS",
                        observations=observations,
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Error fetching BLS data: {e}")
            return []


# =============================================================================
# BEA Tool
# =============================================================================


class BEATool(BaseTool):
    """Tool for accessing Bureau of Economic Analysis data."""

    def __init__(
        self, config: Optional[ToolConfig] = None, econ_config: Optional[EconomicDataConfig] = None
    ):
        super().__init__(config)
        self.econ_config = econ_config or EconomicDataConfig()

    @property
    def name(self) -> str:
        return "bea_data"

    @property
    def description(self) -> str:
        return (
            "Access Bureau of Economic Analysis data including GDP, "
            "personal income, international trade, and regional economic data."
        )

    async def execute(
        self,
        dataset_name: str = "NIPA",
        table_name: str = "T10101",
        frequency: str = "Q",
        year: Optional[str] = None,
    ) -> List[EconomicSeries]:
        """
        Fetch data from BEA API.

        Args:
            dataset_name: BEA dataset (NIPA, Regional, etc.)
            table_name: Table identifier
            frequency: A (annual), Q (quarterly), M (monthly)
            year: Year or range (e.g., "2020" or "2018,2019,2020")

        Returns:
            List of EconomicSeries with observations
        """
        logger.info(f"Fetching BEA data: {dataset_name}/{table_name}")

        if not self.econ_config.bea_api_key:
            logger.warning("BEA API key not configured")
            return []

        current_year = datetime.now().year
        year = year or f"{current_year - 5},{current_year}"

        params = {
            "UserID": self.econ_config.bea_api_key,
            "method": "GetData",
            "DataSetName": dataset_name,
            "TableName": table_name,
            "Frequency": frequency,
            "Year": year,
            "ResultFormat": "JSON",
        }

        try:
            response = await self.http_client.get(
                self.econ_config.bea_base_url,
                params=params,
                timeout=self.econ_config.http_timeout,
            )
            response.raise_for_status()
            data = response.json()

            results = []
            bea_data = data.get("BEAAPI", {}).get("Results", {}).get("Data", [])

            # Group by line description
            series_map: Dict[str, List[EconomicDataPoint]] = {}
            for item in bea_data:
                line_desc = item.get("LineDescription", "Unknown")
                if line_desc not in series_map:
                    series_map[line_desc] = []

                time_period = item.get("TimePeriod", "")
                series_map[line_desc].append(
                    EconomicDataPoint(
                        date=time_period,
                        value=item.get("DataValue", ""),
                        series_id=f"{table_name}_{item.get('LineNumber', '')}",
                        series_name=line_desc,
                        source="BEA",
                        unit=item.get("UNIT_MULT_DESC"),
                    )
                )

            for title, observations in series_map.items():
                results.append(
                    EconomicSeries(
                        series_id=f"{dataset_name}_{table_name}",
                        title=title,
                        source="BEA",
                        frequency=frequency,
                        observations=observations,
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Error fetching BEA data: {e}")
            return []


# =============================================================================
# Census Tool
# =============================================================================


class CensusTool(BaseTool):
    """Tool for accessing U.S. Census Bureau data."""

    def __init__(
        self, config: Optional[ToolConfig] = None, econ_config: Optional[EconomicDataConfig] = None
    ):
        super().__init__(config)
        self.econ_config = econ_config or EconomicDataConfig()

    @property
    def name(self) -> str:
        return "census_data"

    @property
    def description(self) -> str:
        return (
            "Access U.S. Census Bureau data including population, "
            "housing, economic census, and American Community Survey data."
        )

    async def execute(
        self,
        dataset: str = "acs/acs5",
        year: int = 2022,
        variables: List[str] = None,
        geography: str = "us:*",
    ) -> List[EconomicSeries]:
        """
        Fetch data from Census API.

        Args:
            dataset: Census dataset (e.g., 'acs/acs5', 'pep/population')
            year: Data year
            variables: List of variable codes (e.g., ['B01001_001E'])
            geography: Geographic level (e.g., 'us:*', 'state:*')

        Returns:
            List of EconomicSeries with observations
        """
        logger.info(f"Fetching Census data: {dataset}/{year}")

        variables = variables or ["NAME"]

        params = {
            "get": ",".join(variables),
            "for": geography,
        }
        if self.econ_config.census_api_key:
            params["key"] = self.econ_config.census_api_key

        try:
            url = f"{self.econ_config.census_base_url}/{year}/{dataset}"
            response = await self.http_client.get(
                url,
                params=params,
                timeout=self.econ_config.http_timeout,
            )
            response.raise_for_status()
            data = response.json()

            if not data or len(data) < 2:
                return []

            headers = data[0]
            results = []

            for var in variables:
                if var in headers:
                    var_idx = headers.index(var)
                    observations = []

                    for row in data[1:]:
                        observations.append(
                            EconomicDataPoint(
                                date=str(year),
                                value=str(row[var_idx]) if var_idx < len(row) else "",
                                series_id=var,
                                source="Census",
                            )
                        )

                    results.append(
                        EconomicSeries(
                            series_id=var,
                            title=var,
                            source="Census",
                            observations=observations,
                        )
                    )

            return results

        except Exception as e:
            logger.error(f"Error fetching Census data: {e}")
            return []


# =============================================================================
# Treasury Fiscal Data Tool
# =============================================================================


class TreasuryTool(BaseTool):
    """Tool for accessing U.S. Treasury Fiscal Data."""

    def __init__(
        self, config: Optional[ToolConfig] = None, econ_config: Optional[EconomicDataConfig] = None
    ):
        super().__init__(config)
        self.econ_config = econ_config or EconomicDataConfig()

    @property
    def name(self) -> str:
        return "treasury_data"

    @property
    def description(self) -> str:
        return (
            "Access U.S. Treasury Fiscal Data including national debt, "
            "treasury rates, federal spending, and revenue data."
        )

    async def execute(
        self,
        endpoint: str = "v2/accounting/od/debt_to_penny",
        fields: Optional[List[str]] = None,
        filters: Optional[str] = None,
        sort: str = "-record_date",
        page_size: int = 100,
    ) -> List[EconomicSeries]:
        """
        Fetch data from Treasury Fiscal Data API.

        Args:
            endpoint: API endpoint (e.g., 'v2/accounting/od/debt_to_penny')
            fields: Fields to retrieve
            filters: Filter string (e.g., 'record_date:gte:2020-01-01')
            sort: Sort order (prefix with - for descending)
            page_size: Number of records

        Returns:
            List of EconomicSeries with observations
        """
        logger.info(f"Fetching Treasury data: {endpoint}")

        params = {
            "sort": sort,
            "page[size]": page_size,
            "format": "json",
        }
        if fields:
            params["fields"] = ",".join(fields)
        if filters:
            params["filter"] = filters

        try:
            url = f"{self.econ_config.treasury_base_url}/{endpoint}"
            response = await self.http_client.get(
                url,
                params=params,
                timeout=self.econ_config.http_timeout,
            )
            response.raise_for_status()
            data = response.json()

            records = data.get("data", [])
            if not records:
                return []

            # Group by data type
            series_map: Dict[str, List[EconomicDataPoint]] = {}
            for record in records:
                date = record.get("record_date", "")
                for key, value in record.items():
                    if key != "record_date" and value is not None:
                        if key not in series_map:
                            series_map[key] = []
                        series_map[key].append(
                            EconomicDataPoint(
                                date=date,
                                value=str(value),
                                series_id=key,
                                source="Treasury",
                            )
                        )

            results = []
            for series_id, observations in series_map.items():
                results.append(
                    EconomicSeries(
                        series_id=series_id,
                        title=series_id.replace("_", " ").title(),
                        source="Treasury",
                        observations=observations,
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Error fetching Treasury data: {e}")
            return []


# =============================================================================
# Unified Economic Data Tool
# =============================================================================


# Common economic indicators mapped to their series IDs
COMMON_INDICATORS = {
    # FRED indicators
    "gdp": {"source": "fred", "series_id": "GDP", "name": "Gross Domestic Product"},
    "real_gdp": {"source": "fred", "series_id": "GDPC1", "name": "Real GDP"},
    "unemployment": {"source": "fred", "series_id": "UNRATE", "name": "Unemployment Rate"},
    "cpi": {"source": "fred", "series_id": "CPIAUCSL", "name": "Consumer Price Index"},
    "core_cpi": {"source": "fred", "series_id": "CPILFESL", "name": "Core CPI"},
    "fed_funds": {"source": "fred", "series_id": "FEDFUNDS", "name": "Federal Funds Rate"},
    "prime_rate": {"source": "fred", "series_id": "DPRIME", "name": "Prime Rate"},
    "treasury_10y": {"source": "fred", "series_id": "DGS10", "name": "10-Year Treasury Rate"},
    "treasury_2y": {"source": "fred", "series_id": "DGS2", "name": "2-Year Treasury Rate"},
    "sp500": {"source": "fred", "series_id": "SP500", "name": "S&P 500 Index"},
    "housing_starts": {"source": "fred", "series_id": "HOUST", "name": "Housing Starts"},
    "retail_sales": {"source": "fred", "series_id": "RSXFS", "name": "Retail Sales"},
    "industrial_production": {
        "source": "fred",
        "series_id": "INDPRO",
        "name": "Industrial Production",
    },
    "consumer_sentiment": {"source": "fred", "series_id": "UMCSENT", "name": "Consumer Sentiment"},
    "pce": {"source": "fred", "series_id": "PCE", "name": "Personal Consumption Expenditures"},
    "m2": {"source": "fred", "series_id": "M2SL", "name": "M2 Money Supply"},
    # BLS indicators
    "nonfarm_payrolls": {"source": "bls", "series_id": "CES0000000001", "name": "Nonfarm Payrolls"},
    "avg_hourly_earnings": {
        "source": "bls",
        "series_id": "CES0500000003",
        "name": "Average Hourly Earnings",
    },
}


class EconomicDataTool(BaseTool):
    """Unified tool for accessing all economic data sources.

    Sources are automatically enabled/disabled based on API key availability:
    - FRED: Requires FRED_API_KEY
    - BLS: Works without key (limited), better with BLS_API_KEY
    - BEA: Requires BEA_API_KEY
    - Census: Works without key (limited), better with CENSUS_API_KEY
    - Treasury: No API key required (always available)
    """

    def __init__(
        self, config: Optional[ToolConfig] = None, econ_config: Optional[EconomicDataConfig] = None
    ):
        super().__init__(config)
        self.econ_config = econ_config or EconomicDataConfig()
        self.fred = FREDTool(config, self.econ_config)
        self.bls = BLSTool(config, self.econ_config)
        self.bea = BEATool(config, self.econ_config)
        self.census = CensusTool(config, self.econ_config)
        self.treasury = TreasuryTool(config, self.econ_config)

        # Log available sources
        self._log_available_sources()

    def _log_available_sources(self) -> None:
        """Log which data sources are available based on API keys."""
        available = self.get_available_sources()
        logger.info(f"Economic data sources available: {available}")

        if not self.econ_config.fred_api_key:
            logger.warning("FRED_API_KEY not set - FRED data will be unavailable")
        if not self.econ_config.bls_api_key:
            logger.info("BLS_API_KEY not set - BLS will use limited public API")
        if not self.econ_config.bea_api_key:
            logger.warning("BEA_API_KEY not set - BEA data will be unavailable")
        if not self.econ_config.census_api_key:
            logger.info("CENSUS_API_KEY not set - Census will use limited public API")

    def get_available_sources(self) -> List[str]:
        """Get list of available data sources based on API key configuration."""
        sources = ["Treasury"]  # Always available (no API key required)

        if self.econ_config.fred_api_key:
            sources.append("FRED")
        if self.econ_config.bea_api_key:
            sources.append("BEA")

        # BLS and Census work without keys but with limitations
        sources.append("BLS")
        sources.append("Census")

        return sources

    def has_source(self, source: str) -> bool:
        """Check if a specific source is available."""
        return source in self.get_available_sources()

    @property
    def name(self) -> str:
        return "economic_data"

    @property
    def description(self) -> str:
        return (
            "Unified access to economic data from FRED, BLS, BEA, Census, "
            "and Treasury. Retrieves key economic indicators and statistics."
        )

    async def execute(
        self,
        indicators: Optional[List[str]] = None,
        include_gdp: bool = True,
        include_inflation: bool = True,
        include_employment: bool = True,
        include_interest_rates: bool = True,
        include_markets: bool = True,
        include_debt: bool = True,
        start_date: Optional[str] = None,
        limit: int = 50,
    ) -> EconomicDataReport:
        """
        Fetch comprehensive economic data.

        Args:
            indicators: Specific indicators to fetch (uses common names)
            include_gdp: Include GDP data
            include_inflation: Include inflation data
            include_employment: Include employment data
            include_interest_rates: Include interest rate data
            include_markets: Include market data
            include_debt: Include national debt data
            start_date: Start date for observations
            limit: Max observations per series

        Returns:
            EconomicDataReport with all requested data
        """
        logger.info("Fetching comprehensive economic data")

        all_series = []
        all_indicators = []
        sources_used = set()

        # Build list of indicators to fetch
        fred_series = []
        bls_series = []

        if indicators:
            for ind in indicators:
                if ind.lower() in COMMON_INDICATORS:
                    info = COMMON_INDICATORS[ind.lower()]
                    if info["source"] == "fred":
                        fred_series.append(info["series_id"])
                    elif info["source"] == "bls":
                        bls_series.append(info["series_id"])
        else:
            # Default indicators based on flags
            if include_gdp:
                fred_series.extend(["GDP", "GDPC1"])
            if include_inflation:
                fred_series.extend(["CPIAUCSL", "CPILFESL", "PCE"])
            if include_employment:
                fred_series.append("UNRATE")
                bls_series.append("CES0000000001")
            if include_interest_rates:
                fred_series.extend(["FEDFUNDS", "DGS10", "DGS2", "DPRIME"])
            if include_markets:
                fred_series.extend(["SP500", "UMCSENT"])

        # Fetch FRED data (only if API key is available)
        if fred_series and self.econ_config.fred_api_key:
            try:
                fred_data = await self.fred.execute(
                    series_ids=fred_series,
                    start_date=start_date,
                    limit=limit,
                )
                all_series.extend(fred_data)
                sources_used.add("FRED")

                # Create indicators from FRED data
                for series in fred_data:
                    if series.observations:
                        latest = series.observations[0]
                        prev = series.observations[1] if len(series.observations) > 1 else None

                        change = None
                        change_pct = None
                        trend = "stable"

                        if prev and latest.value and prev.value:
                            try:
                                curr_val = float(latest.value)
                                prev_val = float(prev.value)
                                change = f"{curr_val - prev_val:.2f}"
                                if prev_val != 0:
                                    change_pct = f"{((curr_val - prev_val) / prev_val) * 100:.2f}%"
                                trend = (
                                    "rising"
                                    if curr_val > prev_val
                                    else "falling"
                                    if curr_val < prev_val
                                    else "stable"
                                )
                            except ValueError:
                                pass

                        all_indicators.append(
                            EconomicIndicator(
                                name=series.title,
                                current_value=latest.value,
                                previous_value=prev.value if prev else None,
                                change=change,
                                change_percent=change_pct,
                                date=latest.date,
                                source="FRED",
                                trend=trend,
                            )
                        )
            except Exception as e:
                logger.error(f"Error fetching FRED data: {e}")
        elif fred_series:
            logger.info("Skipping FRED data - FRED_API_KEY not configured")

        # Fetch BLS data
        if bls_series:
            try:
                bls_data = await self.bls.execute(series_ids=bls_series)
                all_series.extend(bls_data)
                sources_used.add("BLS")
            except Exception as e:
                logger.error(f"Error fetching BLS data: {e}")

        # Fetch Treasury debt data
        if include_debt:
            try:
                treasury_data = await self.treasury.execute(
                    endpoint="v2/accounting/od/debt_to_penny",
                    page_size=limit,
                )
                all_series.extend(treasury_data)
                sources_used.add("Treasury")

                # Add debt indicator
                for series in treasury_data:
                    if "debt" in series.series_id.lower() and series.observations:
                        latest = series.observations[0]
                        all_indicators.append(
                            EconomicIndicator(
                                name=series.title,
                                current_value=latest.value,
                                date=latest.date,
                                source="Treasury",
                                trend="rising",
                            )
                        )
                        break
            except Exception as e:
                logger.error(f"Error fetching Treasury data: {e}")

        # Generate summary
        summary = self._generate_summary(all_indicators)

        return EconomicDataReport(
            report_date=datetime.now().strftime("%Y-%m-%d"),
            topic="Economic Indicators",
            indicators=all_indicators,
            series_data=all_series,
            summary=summary,
            data_sources=list(sources_used),
        )

    def _generate_summary(self, indicators: List[EconomicIndicator]) -> str:
        """Generate a summary of economic conditions."""
        if not indicators:
            return "No economic data available."

        lines = ["Current Economic Conditions Summary:"]

        for ind in indicators[:10]:
            trend_emoji = "📈" if ind.trend == "rising" else "📉" if ind.trend == "falling" else "➡️"
            change_str = f" ({ind.change_percent})" if ind.change_percent else ""
            lines.append(f"- {ind.name}: {ind.current_value}{change_str} {trend_emoji}")

        return "\n".join(lines)

    async def close(self):
        """Close all tool connections."""
        await super().close()
        await self.fred.close()
        await self.bls.close()
        await self.bea.close()
        await self.census.close()
        await self.treasury.close()


# =============================================================================
# Markdown Formatter
# =============================================================================


def format_economic_data_markdown(report: EconomicDataReport) -> str:
    """Format economic data report as markdown."""
    lines = [
        "## Economic Data Analysis",
        "",
        f"**Report Date:** {report.report_date}",
        f"**Data Sources:** {', '.join(report.data_sources)}",
        "",
        "### Summary",
        "",
        report.summary,
        "",
    ]

    if report.indicators:
        lines.append("### Key Economic Indicators")
        lines.append("")
        lines.append("| Indicator | Current Value | Change | Trend | Date | Source |")
        lines.append("|-----------|---------------|--------|-------|------|--------|")

        for ind in report.indicators:
            trend_emoji = "📈" if ind.trend == "rising" else "📉" if ind.trend == "falling" else "➡️"
            change = ind.change_percent or ind.change or "N/A"
            lines.append(
                f"| {ind.name} | {ind.current_value} | {change} | {trend_emoji} | {ind.date} | {ind.source} |"
            )

        lines.append("")

    return "\n".join(lines)
