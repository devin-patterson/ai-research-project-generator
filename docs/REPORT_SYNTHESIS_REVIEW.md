# Report Synthesis Logic Review & Recommendations

## Executive Summary

This document provides a comprehensive review of the report generation system, identifying gaps in the current implementation and providing actionable recommendations for improving data accuracy, report structure, and output quality.

---

## 1. Current Architecture Analysis

### 1.1 Data Flow Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ResearchRequest                                                             │
│  (topic, research_question, discipline, additional_context)                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  Tool Planning (_plan_tool_execution)                                        │
│  - Analyzes discipline and topic keywords                                    │
│  - Determines which data sources to query                                    │
│  ISSUE: Planning logic is basic keyword matching                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  Parallel Data Collection                                                    │
│  - Economic data (FRED, BLS, IMF, World Bank)                               │
│  - News aggregation (NewsAPI, GNews, RSS)                                   │
│  - Market data (Yahoo, Finnhub, Polygon)                                    │
│  - Industry data (SEC EDGAR, USPTO)                                         │
│  - Social sentiment (Reddit, Twitter)                                       │
│  ISSUE: Data collected is generic, not topic-specific                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  Context Building (_synthesize_collected_data)                               │
│  - Concatenates data into text blocks                                        │
│  - Limited formatting/structure                                              │
│  ISSUE: Context is flat text, loses data relationships                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  LLM Synthesis                                                               │
│  - Single prompt with all context                                            │
│  - Generic system prompt                                                     │
│  ISSUE: No structured output enforcement                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  Markdown Export (_export_markdown)                                          │
│  - Fixed template structure                                                  │
│  - No discipline-specific formatting                                         │
│  ISSUE: One-size-fits-all output format                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Files Reviewed

| File | Purpose | Lines |
|------|---------|-------|
| `research_service.py` | Main orchestration, data collection, synthesis | 1510 |
| `models/schemas/research.py` | Request/Response Pydantic models | 282 |
| `templates/base.py` | Template base class | 226 |
| `templates/investment.py` | Investment research template | 620 |
| `templates/manager.py` | Template registry | 129 |

---

## 2. Identified Issues

### 2.1 Data Collection Issues

#### Issue 1: Generic Data Queries
**Location**: `_collect_imf_data`, `_collect_world_bank_data`, etc.

**Problem**: Data collection uses hardcoded parameters regardless of topic.

```python
# Current implementation (lines 1103-1107)
result = await self._imf_tool.execute(
    indicators=["NGDP_RPCH", "PCPIPCH", "LUR"],  # Always same indicators
    countries=["USA", "CHN", "DEU", "JPN", "GBR"],  # Always same countries
    start_year=2022,
)
```

**Impact**: A healthcare revenue integrity report gets the same IMF GDP data as a Cisco stock analysis.

#### Issue 2: No Topic-Aware Entity Extraction
**Location**: `_collect_sec_data` (lines 1188-1194)

**Problem**: Naive company name extraction from topic.

```python
# Current implementation
for word in topic_words:
    if word[0].isupper() and len(word) > 2:
        company = word
        break
```

**Impact**: "Cisco Systems" becomes "Cisco" but SEC search may not find it. "Revenue Integrity Management" extracts "Revenue" incorrectly.

#### Issue 3: Missing Data Source Correlation
**Problem**: Each data source is queried independently with no cross-referencing.

**Impact**: Stock analysis doesn't correlate SEC filings with news sentiment or market data.

### 2.2 Synthesis Logic Issues

#### Issue 4: Flat Context Structure
**Location**: `_synthesize_collected_data` (lines 1244-1408)

**Problem**: Context is built as flat markdown text with no semantic structure.

```python
# Current approach
context_parts.append(f"- {ind['country']} {ind['name']}: {ind['value']:.2f}% ({ind['year']})")
```

**Impact**: LLM cannot easily identify relationships between data points.

#### Issue 5: Generic Synthesis Prompt
**Location**: Lines 1384-1401

**Problem**: Same prompt template for all research types.

```python
prompt = f"""Based on the following REAL DATA collected from multiple sources...
Generate a detailed report that:
1. Directly answers the research question using the collected data
2. Provides specific recommendations based on the real data
...
"""
```

**Impact**: Investment analysis uses same prompt as healthcare research.

#### Issue 6: No Output Schema Enforcement
**Problem**: LLM output is free-form text, not structured.

**Impact**: Inconsistent report sections, missing required elements.

### 2.3 Schema Issues

#### Issue 7: ResearchResponse Missing Collected Data
**Location**: `models/schemas/research.py` (lines 200-241)

**Problem**: `ResearchResponse` doesn't include `collected_data` field.

```python
class ResearchResponse(BaseModel):
    # ... existing fields
    # MISSING: collected_data: Optional[Dict[str, Any]] = None
```

**Impact**: API consumers can't see what data sources were used.

#### Issue 8: No Structured Report Sections Schema
**Problem**: Report content is stored as raw strings, not structured objects.

```python
ai_direct_research: Optional[str] = None  # Just a string
ai_literature_synthesis: Optional[str] = None  # Just a string
```

**Impact**: Cannot programmatically extract specific sections or validate content.

### 2.4 Template Issues

#### Issue 9: Templates Not Integrated with Data Collection
**Location**: `templates/investment.py`

**Problem**: Investment template defines parameters but doesn't influence data collection.

**Impact**: Template parameters like `asset_classes`, `geographic_focus` aren't used to customize API queries.

#### Issue 10: Single Template Available
**Problem**: Only `InvestmentResearchTemplate` exists.

**Impact**: Healthcare, technology, and other domains use generic processing.

---

## 3. Recommendations

### 3.1 High Priority - Data Quality

#### Recommendation 1: Topic-Aware Data Collection

Create an entity extraction step before data collection:

```python
@dataclass
class TopicEntities:
    """Extracted entities from research topic."""
    companies: List[str]  # ["Cisco Systems", "CSCO"]
    industries: List[str]  # ["networking", "cybersecurity"]
    geographic_regions: List[str]  # ["Florida", "USA"]
    time_periods: List[str]  # ["2024", "Q1 2025"]
    key_metrics: List[str]  # ["revenue", "market share"]
    people: List[str]  # ["Warren Buffett"]

async def _extract_topic_entities(self, request: ResearchRequest) -> TopicEntities:
    """Use LLM to extract structured entities from topic."""
    prompt = f"""Extract key entities from this research topic:
    Topic: {request.topic}
    Question: {request.research_question}
    Context: {request.additional_context}
    
    Return JSON with: companies, industries, geographic_regions, time_periods, key_metrics, people
    """
    # Use structured output with Pydantic model
```

#### Recommendation 2: Dynamic Data Source Configuration

```python
@dataclass
class DataCollectionPlan:
    """Plan for what data to collect based on topic analysis."""
    
    # Economic data config
    imf_indicators: List[str]
    imf_countries: List[str]
    world_bank_category: str
    
    # Market data config
    stock_symbols: List[str]
    include_financials: bool
    include_news: bool
    
    # Industry data config
    sec_companies: List[str]
    sec_form_types: List[str]
    
    # News config
    news_keywords: List[str]
    news_sources: List[str]

def _create_collection_plan(
    self, request: ResearchRequest, entities: TopicEntities
) -> DataCollectionPlan:
    """Create customized data collection plan based on extracted entities."""
```

### 3.2 High Priority - Output Quality

#### Recommendation 3: Structured Report Schema

```python
class ReportSection(BaseModel):
    """A structured report section."""
    title: str
    content: str
    data_sources: List[str]
    confidence_score: float = Field(ge=0.0, le=1.0)
    key_findings: List[str]
    
class ExecutiveSummary(BaseModel):
    """Executive summary with key takeaways."""
    overview: str
    key_findings: List[str]
    recommendations: List[str]
    risks: List[str]
    opportunities: List[str]

class DataEvidence(BaseModel):
    """Evidence from collected data."""
    source: str
    data_point: str
    value: Any
    timestamp: Optional[str]
    relevance: str

class StructuredReport(BaseModel):
    """Fully structured research report."""
    executive_summary: ExecutiveSummary
    methodology: ReportSection
    data_analysis: ReportSection
    findings: List[ReportSection]
    recommendations: ReportSection
    limitations: ReportSection
    appendix: Optional[ReportSection]
    evidence: List[DataEvidence]
    data_sources_used: List[str]
    confidence_metrics: Dict[str, float]
```

#### Recommendation 4: Discipline-Specific Synthesis Prompts

```python
SYNTHESIS_PROMPTS = {
    "finance": """You are a senior financial analyst. Analyze the provided market data, 
    SEC filings, and economic indicators to provide investment recommendations.
    
    Required sections:
    1. Executive Summary with BUY/SELL/HOLD recommendation
    2. Fundamental Analysis (revenue, margins, debt ratios)
    3. Technical Analysis (price trends, volume)
    4. Risk Assessment (market, regulatory, competitive)
    5. Valuation Analysis (P/E, DCF, comparables)
    6. Recommendation with price targets
    
    Always cite specific data points with sources.""",
    
    "healthcare": """You are a healthcare industry analyst. Analyze the provided 
    regulatory data, market trends, and industry reports.
    
    Required sections:
    1. Executive Summary
    2. Regulatory Landscape
    3. Market Analysis
    4. Operational Recommendations
    5. Staffing and Resource Considerations
    6. Technology and Innovation Trends
    7. Risk Mitigation Strategies
    
    Focus on actionable insights for healthcare executives.""",
    
    "technology": """You are a technology research analyst...""",
}
```

### 3.3 Medium Priority - Schema Enhancements

#### Recommendation 5: Add Collected Data to Response

```python
class CollectedDataSummary(BaseModel):
    """Summary of data collected from various sources."""
    sources_queried: List[str]
    sources_successful: List[str]
    sources_failed: List[str]
    data_points_collected: int
    collection_timestamp: datetime
    cache_hits: int
    
class ResearchResponse(BaseModel):
    # ... existing fields
    
    # NEW: Add collected data visibility
    collected_data_summary: Optional[CollectedDataSummary] = None
    raw_collected_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Raw collected data (optional, for debugging)"
    )
```

#### Recommendation 6: Report Quality Metrics

```python
class ReportQualityMetrics(BaseModel):
    """Quality metrics for generated report."""
    data_coverage_score: float  # % of requested data successfully collected
    source_diversity_score: float  # Number of unique sources used
    citation_density: float  # Citations per 100 words
    recommendation_specificity: float  # How specific are recommendations
    evidence_strength: float  # How well supported are claims
    
    overall_quality: float
    improvement_suggestions: List[str]
```

### 3.4 Medium Priority - Template Enhancements

#### Recommendation 7: Create Domain-Specific Templates

```python
# templates/healthcare.py
@dataclass
class HealthcareResearchTemplate(ResearchTemplate):
    template_id: str = "healthcare_research"
    name: str = "Healthcare Industry Research"
    
    @property
    def parameters(self) -> list[TemplateParameter]:
        return [
            TemplateParameter(
                name="healthcare_sector",
                display_name="Healthcare Sector",
                param_type=ParameterType.SELECT,
                options=[
                    "revenue_cycle",
                    "clinical_operations",
                    "health_it",
                    "pharmaceuticals",
                    "medical_devices",
                    "payer_services",
                ],
            ),
            TemplateParameter(
                name="geographic_scope",
                display_name="Geographic Scope",
                param_type=ParameterType.SELECT,
                options=["national", "state", "regional", "facility"],
            ),
            TemplateParameter(
                name="state",
                display_name="State (if applicable)",
                param_type=ParameterType.STRING,
                required=False,
            ),
            # ... more healthcare-specific parameters
        ]
    
    def get_data_collection_config(self, params: dict) -> DataCollectionPlan:
        """Return healthcare-specific data collection configuration."""
        # This method would customize what data sources to query
```

#### Recommendation 8: Template-Driven Data Collection

```python
class ResearchTemplate(ABC):
    # ... existing methods
    
    @abstractmethod
    def get_data_collection_config(self, params: dict[str, Any]) -> DataCollectionPlan:
        """Return configuration for data collection based on template parameters."""
        pass
    
    @abstractmethod
    def get_synthesis_prompt(self, params: dict[str, Any]) -> str:
        """Return discipline-specific synthesis prompt."""
        pass
    
    @abstractmethod
    def get_output_schema(self) -> Type[BaseModel]:
        """Return the expected output schema for this template."""
        pass
```

### 3.5 Low Priority - Advanced Features

#### Recommendation 9: Multi-Stage Synthesis

```python
async def _synthesize_collected_data_staged(
    self, request: ResearchRequest, collected_data: dict, papers: list
) -> StructuredReport:
    """Multi-stage synthesis for higher quality output."""
    
    # Stage 1: Data summarization (per source)
    summaries = {}
    for source, data in collected_data.items():
        summaries[source] = await self._summarize_source(source, data)
    
    # Stage 2: Cross-source analysis
    correlations = await self._analyze_correlations(summaries)
    
    # Stage 3: Generate findings
    findings = await self._generate_findings(summaries, correlations, request)
    
    # Stage 4: Generate recommendations
    recommendations = await self._generate_recommendations(findings, request)
    
    # Stage 5: Compile final report
    return await self._compile_report(
        summaries, correlations, findings, recommendations, request
    )
```

#### Recommendation 10: Confidence Scoring

```python
def _calculate_confidence_score(
    self, collected_data: dict, report_section: str
) -> float:
    """Calculate confidence score for a report section based on data quality."""
    
    factors = {
        "data_recency": self._score_data_recency(collected_data),
        "source_count": self._score_source_diversity(collected_data),
        "data_completeness": self._score_completeness(collected_data),
        "source_reliability": self._score_source_reliability(collected_data),
    }
    
    weights = {
        "data_recency": 0.25,
        "source_count": 0.25,
        "data_completeness": 0.30,
        "source_reliability": 0.20,
    }
    
    return sum(factors[k] * weights[k] for k in factors)
```

---

## 4. Implementation Priority Matrix

| Recommendation | Impact | Effort | Priority |
|----------------|--------|--------|----------|
| 1. Topic-Aware Data Collection | High | Medium | P1 |
| 2. Dynamic Data Source Config | High | Medium | P1 |
| 3. Structured Report Schema | High | Medium | P1 |
| 4. Discipline-Specific Prompts | High | Low | P1 |
| 5. Add Collected Data to Response | Medium | Low | P2 |
| 6. Report Quality Metrics | Medium | Medium | P2 |
| 7. Domain-Specific Templates | Medium | High | P2 |
| 8. Template-Driven Data Collection | Medium | High | P2 |
| 9. Multi-Stage Synthesis | High | High | P3 |
| 10. Confidence Scoring | Medium | Medium | P3 |

---

## 5. Quick Wins (Immediate Implementation)

### 5.1 Add Collected Data to API Response

```python
# In research_service.py, modify generate_project to include collected_data
response = ResearchResponse(
    # ... existing fields
    # Add this:
    collected_data_summary=CollectedDataSummary(
        sources_queried=list(collected_data.keys()),
        sources_successful=[k for k, v in collected_data.items() if not v.get("error")],
        sources_failed=[k for k, v in collected_data.items() if v.get("error")],
        data_points_collected=sum(
            len(v.get("indicators", v.get("articles", v.get("filings", []))))
            for v in collected_data.values()
        ),
        collection_timestamp=datetime.now(),
    ),
)
```

### 5.2 Improve Synthesis Prompt

```python
# Replace generic prompt with discipline-aware prompt
discipline_prompts = {
    "finance": "You are a CFA-certified financial analyst...",
    "healthcare": "You are a healthcare industry consultant...",
    "technology": "You are a technology research analyst...",
    "economics": "You are an economist specializing in...",
}

prompt = discipline_prompts.get(
    request.discipline.lower(), 
    "You are a research analyst..."
)
```

### 5.3 Add Data Source Citations

```python
# In _synthesize_collected_data, track sources used
sources_used = []
for source, data in collected_data.items():
    if data and not data.get("error"):
        sources_used.append(source)

# Add to prompt
prompt += f"\n\nDATA SOURCES USED: {', '.join(sources_used)}"
prompt += "\n\nIMPORTANT: Cite specific data sources when making claims."
```

---

## 6. Conclusion

The current report synthesis system has a solid foundation but lacks:

1. **Topic-aware data collection** - Data queries are generic
2. **Structured output** - Reports are free-form text
3. **Discipline-specific processing** - Same logic for all domains
4. **Data transparency** - Users can't see what data was collected

Implementing the P1 recommendations would significantly improve report quality and accuracy. The template system is well-designed but underutilized - extending it to drive data collection and synthesis would provide the most value.

---

*Review completed: 2026-02-01*
*Reviewer: AI Research Generator Analysis*
