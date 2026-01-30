"""
Research Service Layer

Provides the business logic for research project generation.
Implements the service pattern for clean separation of concerns.
"""

import uuid
from dataclasses import asdict
from datetime import datetime
from typing import Optional, Tuple

from loguru import logger

from app.core.config import Settings
from app.core.exceptions import (
    LLMGenerationError,
    ResearchGenerationError,
)
from app.schemas.research import (
    ResearchRequest,
    ResearchResponse,
    PaperSchema,
    PaperStatistics,
    ValidationReportSchema,
)

# Import existing modules
import sys

sys.path.insert(0, ".")
from llm_provider import LLMConfig, LLMProvider, ResearchLLMAssistant
from academic_search import UnifiedAcademicSearch
from ai_research_project_generator import (
    AIResearchProjectGenerator,
    ResearchContext,
    ResearchType as BaseResearchType,
    AcademicLevel as BaseAcademicLevel,
)
from subject_analyzer import SubjectAnalyzer
from validation_engine import ValidationEngine


class ResearchService:
    """
    Service class for research project generation.

    Encapsulates all business logic and provides a clean interface
    for the API layer to interact with.
    """

    def __init__(self, settings: Settings):
        """
        Initialize the research service with configuration.

        Args:
            settings: Application settings instance
        """
        self.settings = settings
        self._llm_assistant: Optional[ResearchLLMAssistant] = None
        self._academic_search: Optional[UnifiedAcademicSearch] = None
        self._project_generator = AIResearchProjectGenerator()
        self._subject_analyzer = SubjectAnalyzer()
        self._validation_engine = ValidationEngine()

        logger.info("ResearchService initialized")

    async def startup(self) -> None:
        """Initialize resources on application startup"""
        logger.info("Starting ResearchService resources...")

        # Initialize LLM if configured
        try:
            await self._init_llm()
        except Exception as e:
            logger.warning(f"LLM initialization failed: {e}. Continuing without LLM.")

        # Initialize academic search
        self._init_academic_search()

        logger.info("ResearchService startup complete")

    async def shutdown(self) -> None:
        """Cleanup resources on application shutdown"""
        logger.info("Shutting down ResearchService...")

        if self._llm_assistant:
            self._llm_assistant.close()
            self._llm_assistant = None

        if self._academic_search:
            self._academic_search.close()
            self._academic_search = None

        logger.info("ResearchService shutdown complete")

    async def _init_llm(self) -> None:
        """Initialize LLM assistant"""
        config = LLMConfig(
            provider=LLMProvider(self.settings.llm_provider),
            model=self.settings.llm_model,
            base_url=self.settings.llm_base_url,
            api_key=self.settings.llm_api_key,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
            timeout=self.settings.llm_timeout,
        )
        self._llm_assistant = ResearchLLMAssistant(config)
        logger.info(f"LLM initialized: {config.model}")

    def _init_academic_search(self) -> None:
        """Initialize academic search clients"""
        self._academic_search = UnifiedAcademicSearch(
            semantic_scholar_key=self.settings.semantic_scholar_api_key,
            openalex_email=self.settings.openalex_email,
            crossref_email=self.settings.crossref_email,
        )
        logger.info("Academic search initialized")

    @property
    def llm_available(self) -> bool:
        """Check if LLM is available"""
        return self._llm_assistant is not None

    @property
    def academic_search_available(self) -> bool:
        """Check if academic search is available"""
        return self._academic_search is not None

    async def generate_project(self, request: ResearchRequest) -> ResearchResponse:
        """
        Generate a comprehensive research project.

        Args:
            request: Validated research request

        Returns:
            Complete research response with all generated content

        Raises:
            ResearchGenerationError: If generation fails
        """
        request_id = str(uuid.uuid4())
        logger.info(f"Generating project {request_id}: {request.topic[:50]}...")

        try:
            # Step 1: Generate base project structure
            logger.info(f"[{request_id}] Step 1: Generating base project structure")
            base_project, context = self._generate_base_project(request)

            # Step 2: Perform subject analysis
            logger.info(f"[{request_id}] Step 2: Performing subject analysis")
            subject_analysis = self._analyze_subject(request)

            # Step 3: Validate project
            logger.info(f"[{request_id}] Step 3: Validating project")
            validation_report = self._validate_project(base_project, context)

            # Step 4: Search academic papers (if enabled)
            discovered_papers = []
            paper_statistics = None
            search_sources = []

            if request.search_papers and self._academic_search:
                logger.info(f"[{request_id}] Step 4: Searching academic databases")
                discovered_papers, paper_statistics, search_sources = await self._search_papers(
                    request
                )

            # Step 5: Enhance with LLM (if enabled)
            ai_content = {}
            llm_model_used = None

            if request.use_llm and self._llm_assistant:
                logger.info(f"[{request_id}] Step 5: Enhancing with LLM analysis")
                ai_content, llm_model_used = await self._enhance_with_llm(
                    request, discovered_papers
                )

            # Build response
            response = ResearchResponse(
                request_id=request_id,
                generated_at=datetime.now(),
                enhanced=bool(ai_content),
                llm_model_used=llm_model_used,
                topic=request.topic,
                research_question=request.research_question,
                research_type=request.research_type.value,
                academic_level=request.academic_level.value,
                discipline=request.discipline,
                ai_topic_analysis=ai_content.get("topic_analysis"),
                ai_research_questions=ai_content.get("research_questions"),
                ai_methodology_recommendations=ai_content.get("methodology"),
                ai_search_strategy=ai_content.get("search_strategy"),
                ai_literature_synthesis=ai_content.get("synthesis"),
                discovered_papers=[PaperSchema(**p) for p in discovered_papers]
                if discovered_papers
                else None,
                paper_statistics=paper_statistics,
                search_sources_used=search_sources if search_sources else None,
                base_project=base_project,
                subject_analysis=subject_analysis,
                validation_report=validation_report,
            )

            # Generate markdown export if requested
            if request.output_format == "markdown":
                response.markdown_output = self._export_markdown(response)

            logger.info(f"[{request_id}] Project generation complete")
            return response

        except Exception as e:
            logger.error(f"[{request_id}] Generation failed: {e}")
            raise ResearchGenerationError(f"Failed to generate project: {str(e)}")

    def _generate_base_project(self, request: ResearchRequest) -> Tuple[dict, ResearchContext]:
        """Generate base project structure using rule-based generator"""
        context = ResearchContext(
            topic=request.topic,
            research_question=request.research_question,
            research_type=BaseResearchType(request.research_type.value),
            academic_level=BaseAcademicLevel(request.academic_level.value),
            discipline=request.discipline,
            target_publication=request.target_publication,
            citation_style=request.citation_style,
        )

        project = self._project_generator.generate_research_project(context)
        return asdict(project), context

    def _analyze_subject(self, request: ResearchRequest) -> dict:
        """Perform subject analysis"""
        analysis = self._subject_analyzer.analyze_subject(
            request.topic, request.research_question, request.discipline
        )
        return asdict(analysis)

    def _validate_project(
        self, base_project: dict, context: ResearchContext
    ) -> ValidationReportSchema:
        """Validate the generated project"""
        project_data = {
            "context": asdict(context),
            "literature_search": base_project.get("literature_search", {}),
            "methodology": base_project.get("methodology", {}),
            "research_questions": base_project.get("research_questions", []),
            "timeline": base_project.get("timeline", {}),
            "expected_outcomes": base_project.get("expected_outcomes", []),
        }

        report = self._validation_engine.validate_project(project_data)

        return ValidationReportSchema(
            overall_score=report.overall_score,
            issues=[
                {
                    "level": i.level.value,
                    "title": i.title,
                    "description": i.description,
                    "category": i.category.value,
                }
                for i in report.issues
            ],
            recommendations=report.recommendations,
            enhancement_opportunities=report.enhancement_opportunities,
        )

    async def _search_papers(
        self, request: ResearchRequest
    ) -> Tuple[list, Optional[PaperStatistics], list]:
        """Search academic databases for relevant papers"""
        try:
            year_range = None
            if request.year_start and request.year_end:
                year_range = (request.year_start, request.year_end)
            elif request.year_start:
                year_range = (request.year_start, datetime.now().year)
            elif request.year_end:
                year_range = (2000, request.year_end)
            else:
                # Default to last 5 years
                current_year = datetime.now().year
                year_range = (current_year - 5, current_year)

            papers = self._academic_search.search_merged(
                query=request.topic,
                limit=request.paper_limit,
                year_range=year_range,
                sources=["openalex", "crossref"],  # Skip semantic scholar due to rate limits
            )

            # Convert to dict format
            papers_dict = []
            for p in papers:
                papers_dict.append(
                    {
                        "title": p.title,
                        "authors": p.authors,
                        "abstract": p.abstract,
                        "year": p.year,
                        "doi": p.doi,
                        "url": p.url,
                        "citation_count": p.citation_count,
                        "venue": p.venue,
                        "source": p.source,
                        "pdf_url": p.pdf_url,
                    }
                )

            # Calculate statistics
            stats = self._calculate_paper_stats(papers)

            return papers_dict, stats, ["openalex", "crossref"]

        except Exception as e:
            logger.error(f"Academic search error: {e}")
            return [], None, []

    def _calculate_paper_stats(self, papers: list) -> Optional[PaperStatistics]:
        """Calculate statistics about discovered papers"""
        if not papers:
            return None

        years = [p.year for p in papers if p.year]
        citations = [p.citation_count for p in papers if p.citation_count]
        sources = [p.source for p in papers]

        return PaperStatistics(
            total_papers=len(papers),
            papers_with_abstracts=sum(1 for p in papers if p.abstract),
            papers_with_pdf=sum(1 for p in papers if p.pdf_url),
            year_range={
                "min": min(years) if years else None,
                "max": max(years) if years else None,
            },
            citation_stats={
                "total": sum(citations) if citations else 0,
                "average": sum(citations) / len(citations) if citations else 0,
                "max": max(citations) if citations else 0,
            },
            sources={source: sources.count(source) for source in set(sources)},
        )

    async def _enhance_with_llm(self, request: ResearchRequest, papers: list) -> Tuple[dict, str]:
        """Enhance project with LLM-generated content"""
        content = {}
        model_used = None

        try:
            # Topic analysis
            result = self._llm_assistant.analyze_topic(request.topic, request.discipline)
            content["topic_analysis"] = result["analysis"]
            model_used = result.get("model")

            # Research questions
            questions = self._llm_assistant.generate_research_questions(
                request.topic, request.research_type.value, request.research_question
            )
            content["research_questions"] = questions

            # Methodology recommendations
            method_result = self._llm_assistant.recommend_methodology(
                request.topic, request.research_question, request.discipline
            )
            content["methodology"] = method_result["recommendations"]

            # Search strategy
            search_result = self._llm_assistant.generate_search_strategy(
                request.topic, request.discipline
            )
            content["search_strategy"] = search_result["strategy"]

            # Literature synthesis (if papers found)
            if papers:
                paper_dicts = [
                    {
                        "title": p["title"],
                        "authors": ", ".join(p["authors"][:3]),
                        "abstract": p.get("abstract", ""),
                    }
                    for p in papers[:10]
                ]
                synthesis = self._llm_assistant.synthesize_findings(
                    paper_dicts, request.research_question
                )
                content["synthesis"] = synthesis

        except Exception as e:
            logger.error(f"LLM enhancement error: {e}")
            raise LLMGenerationError(f"LLM enhancement failed: {str(e)}")

        return content, model_used

    def _export_markdown(self, response: ResearchResponse) -> str:
        """Export response as markdown"""
        md = f"""# AI-Enhanced Research Project

## Project Overview

**Topic**: {response.topic}  
**Research Question**: {response.research_question}  
**Research Type**: {response.research_type}  
**Academic Level**: {response.academic_level}  
**Discipline**: {response.discipline}  
**Generated**: {response.generated_at.isoformat()}  
**AI Enhanced**: {"Yes" if response.enhanced else "No"}  
**LLM Model**: {response.llm_model_used or "N/A"}  
**Request ID**: {response.request_id}

---

"""

        if response.ai_topic_analysis:
            md += f"""## 🤖 AI Topic Analysis

{response.ai_topic_analysis}

---

"""

        if response.ai_research_questions:
            md += """## 🤖 AI-Generated Research Questions

"""
            for i, q in enumerate(response.ai_research_questions, 1):
                md += f"{i}. {q}\n"
            md += "\n---\n\n"

        if response.ai_methodology_recommendations:
            md += f"""## 🤖 AI Methodology Recommendations

{response.ai_methodology_recommendations}

---

"""

        if response.discovered_papers:
            stats = response.paper_statistics
            md += f"""## 📚 Discovered Papers

**Total Papers Found**: {stats.total_papers if stats else len(response.discovered_papers)}  
**Papers with Abstracts**: {stats.papers_with_abstracts if stats else "N/A"}  
**Sources**: {", ".join(f"{k}: {v}" for k, v in (stats.sources.items() if stats else {}))}

### Top Papers

"""
            for i, paper in enumerate(response.discovered_papers[:10], 1):
                authors = ", ".join(paper.authors[:3]) if paper.authors else "Unknown"
                md += f"""**{i}. {paper.title}**  
Authors: {authors}  
Year: {paper.year or "N/A"} | Citations: {paper.citation_count or "N/A"} | Source: {paper.source}  
"""
                if paper.doi:
                    md += f"DOI: https://doi.org/{paper.doi}  \n"
                md += "\n"

            md += "---\n\n"

        if response.ai_literature_synthesis:
            md += f"""## 🤖 AI Literature Synthesis

{response.ai_literature_synthesis}

---

"""

        md += f"""## ✅ Validation Report

**Quality Score**: {response.validation_report.overall_score:.2f}/1.00

### Recommendations

"""
        for rec in response.validation_report.recommendations[:5]:
            md += f"- {rec}\n"

        md += "\n---\n\n*Generated by AI Research Project Generator API*\n"

        return md
