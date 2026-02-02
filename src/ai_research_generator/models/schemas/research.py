"""
Pydantic schemas for research project API

Defines request/response models with validation for the research generation API.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ResearchType(str, Enum):
    """Supported research methodologies"""

    SYSTEMATIC_REVIEW = "systematic_review"
    SCOPING_REVIEW = "scoping_review"
    META_ANALYSIS = "meta_analysis"
    QUALITATIVE_STUDY = "qualitative_study"
    QUANTITATIVE_STUDY = "quantitative_study"
    MIXED_METHODS = "mixed_methods"
    CASE_STUDY = "case_study"
    EXPERIMENTAL = "experimental"
    LITERATURE_REVIEW = "literature_review"


class AcademicLevel(str, Enum):
    """Academic levels for research projects"""

    UNDERGRADUATE = "undergraduate"
    GRADUATE = "graduate"
    DOCTORAL = "doctoral"
    POST_DOCTORAL = "post_doctoral"
    PROFESSIONAL = "professional"


class LLMProvider(str, Enum):
    """Supported LLM providers"""

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local_openai_compatible"


class ResearchRequest(BaseModel):
    """
    Request schema for generating a research project.

    This is the primary input interface for the research generation API.
    """

    topic: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="The research topic or subject area to investigate",
        examples=["Impact of remote work on employee productivity and well-being"],
    )

    research_question: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="The primary research question to address",
        examples=[
            "How has the shift to remote work affected employee productivity and mental health outcomes?"
        ],
    )

    research_type: ResearchType = Field(
        default=ResearchType.SYSTEMATIC_REVIEW,
        description="The type of research methodology to use",
    )

    academic_level: AcademicLevel = Field(
        default=AcademicLevel.GRADUATE, description="The academic level for the research project"
    )

    discipline: str = Field(
        default="general",
        min_length=2,
        max_length=100,
        description="The academic discipline or field of study",
        examples=["psychology", "computer science", "medicine", "education"],
    )

    # Optional configuration
    target_publication: Optional[str] = Field(
        default=None, max_length=200, description="Target journal or publication venue"
    )

    citation_style: str = Field(
        default="APA", description="Citation style to use (APA, MLA, Chicago, etc.)"
    )

    # Search configuration
    search_papers: bool = Field(
        default=True, description="Whether to search academic databases for relevant papers"
    )

    paper_limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of papers to retrieve from academic search",
    )

    year_start: Optional[int] = Field(
        default=None, ge=1900, le=2030, description="Start year for paper search range"
    )

    year_end: Optional[int] = Field(
        default=None, ge=1900, le=2030, description="End year for paper search range"
    )

    # LLM configuration
    use_llm: bool = Field(default=True, description="Whether to use LLM for enhanced analysis")

    llm_model: str = Field(default="llama3.1:8b", description="LLM model to use for analysis")

    # Additional context for research
    additional_context: Optional[str] = Field(
        default=None,
        max_length=5000,
        description="Additional context, requirements, or focus areas for the research",
    )

    # Output configuration
    output_format: str = Field(
        default="markdown", description="Output format (markdown, json, html)"
    )

    @field_validator("year_end")
    @classmethod
    def validate_year_range(cls, v: Optional[int], info) -> Optional[int]:
        """Ensure year_end is after year_start"""
        if v is not None and info.data.get("year_start") is not None:
            if v < info.data["year_start"]:
                raise ValueError("year_end must be greater than or equal to year_start")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "topic": "Impact of remote work on employee productivity and well-being",
                    "research_question": "How has the shift to remote work affected employee productivity and mental health outcomes?",
                    "research_type": "systematic_review",
                    "academic_level": "graduate",
                    "discipline": "psychology",
                    "search_papers": True,
                    "paper_limit": 20,
                    "year_start": 2020,
                    "year_end": 2026,
                    "use_llm": True,
                    "llm_model": "llama3.1:8b",
                }
            ]
        }
    }


class PaperSchema(BaseModel):
    """Schema for academic paper metadata"""

    title: str
    authors: List[str]
    abstract: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    citation_count: Optional[int] = None
    venue: Optional[str] = None
    source: str
    pdf_url: Optional[str] = None


class ValidationReportSchema(BaseModel):
    """Schema for validation report"""

    overall_score: float = Field(ge=0.0, le=1.0)
    issues: List[Dict[str, Any]] = []
    recommendations: List[str] = []
    enhancement_opportunities: List[str] = []


class PaperStatistics(BaseModel):
    """Statistics about discovered papers"""

    total_papers: int
    papers_with_abstracts: int
    papers_with_pdf: int
    year_range: Dict[str, Optional[int]]
    citation_stats: Dict[str, float]
    sources: Dict[str, int]


class CollectedDataSummary(BaseModel):
    """Summary of data collected from various sources during research."""

    sources_queried: List[str] = Field(
        default_factory=list,
        description="List of all data sources that were queried"
    )
    sources_successful: List[str] = Field(
        default_factory=list,
        description="List of data sources that returned data successfully"
    )
    sources_failed: List[str] = Field(
        default_factory=list,
        description="List of data sources that failed or returned errors"
    )
    data_points_collected: int = Field(
        default=0,
        description="Total number of data points collected across all sources"
    )
    collection_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when data collection completed"
    )
    cache_hits: int = Field(
        default=0,
        description="Number of data points retrieved from cache"
    )
    collection_duration_seconds: Optional[float] = Field(
        default=None,
        description="Time taken for data collection in seconds"
    )


class ExtractedEntities(BaseModel):
    """Entities extracted from research topic for targeted data collection."""

    companies: List[str] = Field(
        default_factory=list,
        description="Company names and stock tickers identified"
    )
    industries: List[str] = Field(
        default_factory=list,
        description="Industry sectors identified"
    )
    geographic_regions: List[str] = Field(
        default_factory=list,
        description="Geographic regions (countries, states, cities)"
    )
    time_periods: List[str] = Field(
        default_factory=list,
        description="Time periods mentioned (years, quarters, dates)"
    )
    key_metrics: List[str] = Field(
        default_factory=list,
        description="Key metrics or KPIs to track"
    )
    people: List[str] = Field(
        default_factory=list,
        description="Notable people mentioned"
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Key search terms for data collection"
    )


class ResearchResponse(BaseModel):
    """
    Response schema for generated research project.

    Contains the complete enhanced research project with AI analysis and papers.
    """

    # Metadata
    request_id: str = Field(description="Unique identifier for this request")
    generated_at: datetime = Field(description="Timestamp when project was generated")
    enhanced: bool = Field(description="Whether LLM enhancement was applied")
    llm_model_used: Optional[str] = Field(description="LLM model used for enhancement")

    # Core project data
    topic: str
    research_question: str
    research_type: str
    academic_level: str
    discipline: str

    # AI-generated content
    ai_topic_analysis: Optional[str] = None
    ai_research_questions: Optional[List[str]] = None
    ai_methodology_recommendations: Optional[str] = None
    ai_search_strategy: Optional[str] = None
    ai_direct_research: Optional[str] = None
    ai_literature_synthesis: Optional[str] = None

    # Discovered papers
    discovered_papers: Optional[List[PaperSchema]] = None
    paper_statistics: Optional[PaperStatistics] = None
    search_sources_used: Optional[List[str]] = None

    # Project structure
    base_project: Dict[str, Any]
    subject_analysis: Dict[str, Any]
    validation_report: ValidationReportSchema

    # Export
    markdown_output: Optional[str] = Field(
        default=None, description="Full project exported as markdown"
    )

    # Data collection metadata
    collected_data_summary: Optional[CollectedDataSummary] = Field(
        default=None,
        description="Summary of data sources queried and data collected"
    )
    extracted_entities: Optional[ExtractedEntities] = Field(
        default=None,
        description="Entities extracted from topic for targeted data collection"
    )


class ProjectStatusResponse(BaseModel):
    """Response for async project status check"""

    request_id: str
    status: str = Field(description="pending, processing, completed, failed")
    progress: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    message: Optional[str] = None
    result: Optional[ResearchResponse] = None


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    version: str
    llm_available: bool
    llm_model: Optional[str] = None
    academic_search_available: bool
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Standard error response"""

    detail: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class LLMConfigRequest(BaseModel):
    """Request to configure LLM settings"""

    provider: LLMProvider = LLMProvider.OLLAMA
    model: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=100, le=32000)
