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

from ..core.config import Settings
from ..core.exceptions import (
    LLMGenerationError,
    ResearchGenerationError,
)
from ..models.schemas.research import (
    ResearchRequest,
    ResearchResponse,
    PaperSchema,
    PaperStatistics,
    ValidationReportSchema,
)

# Import existing modules
from ..legacy.llm_provider import LLMConfig, LLMProvider, ResearchLLMAssistant
from ..legacy.academic_search import UnifiedAcademicSearch
from ..legacy.ai_research_project_generator import (
    AIResearchProjectGenerator,
    ResearchContext,
    ResearchType as BaseResearchType,
    AcademicLevel as BaseAcademicLevel,
)
from ..legacy.subject_analyzer import SubjectAnalyzer
from ..legacy.validation_engine import ValidationEngine

# Import new research tools
from ..tools.research_tools import (
    ResearchToolkit,
    ToolConfig,
    WebSearchTool,
    KnowledgeSynthesisTool,
)
from ..tools.economic_data import EconomicDataTool, EconomicDataConfig
from ..tools.current_events import CurrentEventsTool, CurrentEventsConfig

# RAG and Caching
from ..rag import RAGService, RAGConfig
from ..cache import DataCache, DataCacheConfig


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

        # Initialize new research toolkit
        self._research_toolkit: Optional[ResearchToolkit] = None

        # Initialize data collection tools
        self._economic_tool: Optional[EconomicDataTool] = None
        self._current_events_tool: Optional[CurrentEventsTool] = None
        self._web_search_tool: Optional[WebSearchTool] = None
        self._knowledge_synth_tool: Optional[KnowledgeSynthesisTool] = None

        # RAG and Caching services
        self._rag_service: Optional[RAGService] = None
        self._data_cache: Optional[DataCache] = None

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

        # Initialize research toolkit
        self._init_research_toolkit()

        # Initialize RAG and caching services
        self._init_rag_service()
        self._init_data_cache()

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

        if self._research_toolkit:
            await self._research_toolkit.close()
            self._research_toolkit = None

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

    def _init_research_toolkit(self) -> None:
        """Initialize research toolkit with all tools"""
        tool_config = ToolConfig(
            llm_base_url=self.settings.llm_base_url,
            llm_model=self.settings.llm_model,
        )
        self._research_toolkit = ResearchToolkit(tool_config)

        # Initialize individual data collection tools
        self._economic_tool = EconomicDataTool(tool_config, EconomicDataConfig())
        self._current_events_tool = CurrentEventsTool(tool_config, CurrentEventsConfig())
        self._web_search_tool = WebSearchTool(tool_config)
        self._knowledge_synth_tool = KnowledgeSynthesisTool(tool_config)

        logger.info("Research toolkit initialized")
        logger.info(
            f"Economic data sources available: {self._economic_tool.get_available_sources()}"
        )

    def _init_rag_service(self) -> None:
        """Initialize RAG service with LangChain components."""
        try:
            rag_config = RAGConfig(
                ollama_base_url=self.settings.llm_base_url,
                chunk_size=1000,
                chunk_overlap=200,
            )
            self._rag_service = RAGService(rag_config)
            if self._rag_service.is_available:
                logger.info("RAG service initialized successfully")
                stats = self._rag_service.get_collection_stats()
                logger.info(f"RAG collection stats: {stats}")
            else:
                logger.warning("RAG service initialized but not fully available")
        except Exception as e:
            logger.warning(f"RAG service initialization failed: {e}. Continuing without RAG.")
            self._rag_service = None

    def _init_data_cache(self) -> None:
        """Initialize data cache for API responses."""
        try:
            cache_config = DataCacheConfig(
                ttl_economic_data=14400,  # 4 hours
                ttl_current_events=3600,  # 1 hour
                ttl_academic_papers=86400,  # 24 hours
            )
            self._data_cache = DataCache(cache_config)
            logger.info("Data cache initialized")
        except Exception as e:
            logger.warning(f"Data cache initialization failed: {e}. Continuing without cache.")
            self._data_cache = None

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
            # Step 0: Check semantic cache for similar queries
            cached_report = None
            if self._rag_service and self._rag_service.is_available:
                cache_query = f"{request.topic} | {request.research_question}"
                cached = self._rag_service.cache_lookup(cache_query, threshold=0.90)
                if cached:
                    logger.info(f"[{request_id}] Semantic cache HIT! Score: {cached['score']:.3f}")
                    cached_report = cached["response"]

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

            if cached_report:
                # Use cached report instead of generating new one
                logger.info(f"[{request_id}] Step 5: Using cached research report")
                ai_content["direct_research"] = cached_report
                llm_model_used = "cached"
            elif request.use_llm and self._llm_assistant:
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
                ai_direct_research=ai_content.get("direct_research"),
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

            # Cache the research report for semantic retrieval
            if self._rag_service and self._rag_service.is_available:
                cache_query = f"{request.topic} | {request.research_question}"
                cache_response = response.ai_direct_research or ""
                if cache_response:
                    self._rag_service.cache_store(
                        cache_query,
                        cache_response,
                        metadata={
                            "request_id": request_id,
                            "topic": request.topic,
                            "discipline": request.discipline,
                            "research_type": request.research_type.value,
                        },
                    )
                    logger.info(f"[{request_id}] Cached research report for semantic retrieval")

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

            # Ingest papers into RAG for future retrieval
            if self._rag_service and self._rag_service.is_available and papers_dict:
                for paper in papers_dict[:10]:
                    authors = paper.get("authors") or []
                    authors_str = ", ".join(authors[:3]) if authors else "Unknown"
                    text = (
                        f"Academic Paper: {paper['title']}\n"
                        f"Authors: {authors_str}\n"
                        f"Year: {paper.get('year', 'N/A')}\n"
                        f"Abstract: {paper.get('abstract', 'No abstract available')[:500]}\n"
                        f"DOI: {paper.get('doi', 'N/A')}"
                    )
                    self._rag_service.ingest_text(
                        text,
                        metadata={
                            "type": "academic_paper",
                            "title": paper["title"],
                            "year": paper.get("year"),
                            "doi": paper.get("doi"),
                            "source": paper.get("source"),
                        },
                        collection="academic_papers",
                    )
                logger.info(f"Ingested {min(len(papers_dict), 10)} academic papers into RAG")

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
        """
        Enhance project with LLM-generated content.

        This method now follows a data-driven approach:
        1. Plan what data/tools are needed based on the prompt
        2. Execute tools to gather real data
        3. Synthesize the real data into the final report
        """
        content = {}
        model_used = None
        collected_data = {}

        try:
            # Step 1: Analyze topic to understand data needs
            result = self._llm_assistant.analyze_topic(request.topic, request.discipline)
            content["topic_analysis"] = result["analysis"]
            model_used = result.get("model")

            # Step 2: Plan and execute data collection based on topic
            logger.info("Planning data collection based on prompt analysis...")
            tool_plan = self._plan_tool_execution(request)

            # Step 3: Execute planned tools to gather REAL data
            if tool_plan.get("needs_economic_data") and self._economic_tool:
                logger.info("Executing economic data collection...")
                collected_data["economic"] = await self._collect_economic_data(request)

            if tool_plan.get("needs_current_events") and self._current_events_tool:
                logger.info("Executing current events research...")
                collected_data["current_events"] = await self._collect_current_events(request)

            if tool_plan.get("needs_web_search") and self._web_search_tool:
                logger.info("Executing web search...")
                collected_data["web_search"] = await self._collect_web_data(request)

            # Step 4: Generate research questions
            questions = self._llm_assistant.generate_research_questions(
                request.topic, request.research_type.value, request.research_question
            )
            content["research_questions"] = questions

            # Step 5: Generate methodology (only if no real data collected)
            if not collected_data:
                method_result = self._llm_assistant.recommend_methodology(
                    request.topic, request.research_question, request.discipline
                )
                content["methodology"] = method_result["recommendations"]

            # Step 6: Generate search strategy
            search_result = self._llm_assistant.generate_search_strategy(
                request.topic, request.discipline
            )
            content["search_strategy"] = search_result["strategy"]

            # Step 7: Generate report based on COLLECTED DATA
            if collected_data:
                logger.info(
                    f"Synthesizing report from collected data: {list(collected_data.keys())}"
                )
                direct_research = await self._synthesize_collected_data(
                    request, collected_data, papers
                )
            else:
                # Fallback to LLM-only generation if no tools available
                direct_research = self._llm_assistant.generate_direct_research(
                    request.topic, request.research_question, request.additional_context or ""
                )
            content["direct_research"] = direct_research

            # Step 8: Literature synthesis (if papers found)
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

            # Store collected data in content for reference
            content["collected_data"] = collected_data

        except Exception as e:
            logger.error(f"LLM enhancement error: {e}")
            raise LLMGenerationError(f"LLM enhancement failed: {str(e)}")

        return content, model_used

    def _plan_tool_execution(self, request: ResearchRequest) -> dict:
        """
        Analyze the prompt and determine which tools/data sources are needed.

        Returns a plan dict indicating what data to collect.
        """
        topic_lower = request.topic.lower()
        question_lower = request.research_question.lower()
        combined = f"{topic_lower} {question_lower}"

        plan = {
            "needs_economic_data": False,
            "needs_current_events": False,
            "needs_web_search": False,
            "queries": [],
        }

        # Check for economic/financial data needs
        economic_keywords = [
            "investment",
            "stock",
            "bond",
            "etf",
            "reit",
            "portfolio",
            "retirement",
            "financial",
            "market",
            "economy",
            "gdp",
            "inflation",
            "interest rate",
            "federal reserve",
            "monetary",
            "fiscal",
            "treasury",
            "yield",
            "return",
            "asset allocation",
            "diversification",
            "risk",
            "wealth",
            "savings",
        ]
        if any(kw in combined for kw in economic_keywords):
            plan["needs_economic_data"] = True
            plan["needs_current_events"] = True

        # Check for current events/news needs
        current_keywords = [
            "current",
            "recent",
            "latest",
            "today",
            "2024",
            "2025",
            "2026",
            "trend",
            "outlook",
            "forecast",
            "prediction",
            "news",
            "update",
        ]
        if any(kw in combined for kw in current_keywords):
            plan["needs_current_events"] = True

        # Check for web search needs
        web_keywords = [
            "how to",
            "best practices",
            "guide",
            "tutorial",
            "comparison",
            "review",
            "recommendation",
            "strategy",
            "approach",
        ]
        if any(kw in combined for kw in web_keywords):
            plan["needs_web_search"] = True

        # Generate search queries from the topic
        plan["queries"] = [
            request.topic,
            request.research_question,
            f"{request.topic} {request.discipline}",
        ]

        logger.info(f"Tool execution plan: {plan}")
        return plan

    async def _collect_economic_data(self, request: ResearchRequest) -> dict:
        """Collect economic data from government APIs with caching."""
        cache_key = "economic_data_comprehensive"

        # Check cache first
        if self._data_cache:
            cached = self._data_cache.get(cache_key)
            if cached:
                logger.info("Using cached economic data")
                return cached

        try:
            report = await self._economic_tool.execute(
                include_gdp=True,
                include_inflation=True,
                include_employment=True,
                include_interest_rates=True,
                include_markets=True,
                include_debt=True,
                limit=20,
            )

            result = {
                "report_date": report.report_date,
                "data_sources": report.data_sources,
                "summary": report.summary,
                "indicators": [
                    {
                        "name": ind.name,
                        "value": ind.current_value,
                        "trend": ind.trend,
                        "change": ind.change_percent,
                        "source": ind.source,
                    }
                    for ind in report.indicators
                ],
            }

            # Cache the result
            if self._data_cache:
                self._data_cache.set(cache_key, result, data_type="economic_data")
                logger.info("Cached economic data")

            # Ingest into RAG for future retrieval
            if self._rag_service and self._rag_service.is_available:
                for ind in result["indicators"][:10]:
                    text = f"Economic Indicator: {ind['name']}\nValue: {ind['value']}\nTrend: {ind['trend']}"
                    self._rag_service.ingest_text(
                        text,
                        metadata={"type": "economic_indicator", "source": ind.get("source")},
                        collection="economic_data",
                    )

            return result
        except Exception as e:
            logger.warning(f"Economic data collection failed: {e}")
            return {"error": str(e), "indicators": []}

    async def _collect_current_events(self, request: ResearchRequest) -> dict:
        """Collect current events and market news with caching and RAG ingestion."""
        cache_key = f"current_events_{request.topic[:50]}"

        # Check cache first
        if self._data_cache:
            cached = self._data_cache.get(cache_key)
            if cached:
                logger.info("Using cached current events data")
                return cached

        try:
            analysis = await self._current_events_tool.execute(
                topic=request.topic,
                time_range_days=30,
            )

            result = {
                "topic": analysis.topic,
                "analysis_date": analysis.analysis_date,
                "executive_summary": analysis.executive_summary,
                "events": [
                    {
                        "title": e.title,
                        "summary": e.summary,
                        "date": e.date,
                        "relevance": e.relevance,
                    }
                    for e in analysis.major_events[:10]
                ],
                "market_conditions": [
                    {
                        "indicator": m.indicator,
                        "status": m.current_state,
                        "trend": m.trend,
                        "impact": m.implications,
                    }
                    for m in analysis.market_conditions
                ],
                "risks": analysis.current_risks,
                "opportunities": analysis.current_opportunities,
            }

            # Cache the result
            if self._data_cache:
                self._data_cache.set(cache_key, result, data_type="current_events")
                logger.info("Cached current events data")

            # Ingest into RAG for future retrieval
            if self._rag_service and self._rag_service.is_available:
                # Ingest events
                for event in result["events"][:5]:
                    text = f"Current Event: {event['title']}\nDate: {event['date']}\nSummary: {event['summary']}\nRelevance: {event['relevance']}"
                    self._rag_service.ingest_text(
                        text,
                        metadata={"type": "current_event", "topic": request.topic},
                        collection="current_events",
                    )
                # Ingest market conditions
                for mc in result["market_conditions"][:5]:
                    text = f"Market Condition: {mc['indicator']}\nStatus: {mc['status']}\nTrend: {mc['trend']}\nImpact: {mc['impact']}"
                    self._rag_service.ingest_text(
                        text,
                        metadata={"type": "market_condition", "topic": request.topic},
                        collection="current_events",
                    )
                logger.info("Ingested current events into RAG")

            return result
        except Exception as e:
            logger.warning(f"Current events collection failed: {e}")
            return {"error": str(e), "events": [], "market_conditions": []}

    async def _collect_web_data(self, request: ResearchRequest) -> dict:
        """Collect web search results with caching and RAG ingestion."""
        cache_key = f"web_search_{request.topic[:50]}"

        # Check cache first
        if self._data_cache:
            cached = self._data_cache.get(cache_key)
            if cached:
                logger.info("Using cached web search data")
                return cached

        try:
            results = await self._web_search_tool.execute(
                query=request.topic,
                num_results=10,
            )

            result = {
                "query": request.topic,
                "results": [
                    {
                        "title": r.title,
                        "url": r.url,
                        "snippet": r.snippet,
                        "source": r.source,
                    }
                    for r in results
                ],
            }

            # Cache the result
            if self._data_cache:
                self._data_cache.set(cache_key, result, data_type="web_search")
                logger.info("Cached web search data")

            # Ingest into RAG for future retrieval
            if self._rag_service and self._rag_service.is_available:
                for r in result["results"][:5]:
                    text = f"Web Result: {r['title']}\nSource: {r['source']}\nURL: {r['url']}\nSnippet: {r['snippet']}"
                    self._rag_service.ingest_text(
                        text,
                        metadata={"type": "web_search", "url": r["url"], "source": r["source"]},
                        collection="research_history",
                    )
                logger.info("Ingested web search results into RAG")

            return result
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            return {"error": str(e), "results": []}

    async def _synthesize_collected_data(
        self, request: ResearchRequest, collected_data: dict, papers: list
    ) -> str:
        """
        Synthesize all collected data into a comprehensive report.

        This uses the LLM to analyze REAL data and produce actionable insights.
        """
        # Build context from collected data
        context_parts = []

        # Add economic data context
        if "economic" in collected_data and collected_data["economic"].get("indicators"):
            econ = collected_data["economic"]
            context_parts.append("## Current Economic Data (Real-Time)")
            context_parts.append(f"Data Sources: {', '.join(econ.get('data_sources', []))}")
            context_parts.append(f"Summary: {econ.get('summary', 'N/A')}")
            context_parts.append("\nKey Indicators:")
            for ind in econ.get("indicators", [])[:10]:
                context_parts.append(
                    f"- {ind['name']}: {ind['value']} ({ind['trend']}, {ind.get('change', 'N/A')})"
                )

        # Add current events context
        if "current_events" in collected_data:
            events = collected_data["current_events"]
            if events.get("executive_summary"):
                context_parts.append("\n## Current Market Conditions (Real-Time)")
                context_parts.append(events["executive_summary"])

            if events.get("market_conditions"):
                context_parts.append("\nMarket Indicators:")
                for m in events.get("market_conditions", []):
                    context_parts.append(f"- {m['indicator']}: {m['status']} ({m['trend']})")

            if events.get("risks"):
                context_parts.append(f"\nIdentified Risks: {', '.join(events['risks'][:5])}")

            if events.get("opportunities"):
                context_parts.append(f"Opportunities: {', '.join(events['opportunities'][:5])}")

        # Add web search context
        if "web_search" in collected_data and collected_data["web_search"].get("results"):
            context_parts.append("\n## Web Research Findings")
            for r in collected_data["web_search"].get("results", [])[:5]:
                context_parts.append(f"- {r['title']}: {r['snippet'][:200]}...")

        # Add academic papers context
        if papers:
            context_parts.append(f"\n## Academic Research ({len(papers)} papers found)")
            for p in papers[:5]:
                context_parts.append(f"- {p['title']} ({p.get('year', 'N/A')})")

        collected_context = "\n".join(context_parts)

        # Generate synthesis using LLM with real data
        prompt = f"""Based on the following REAL DATA collected from multiple sources, provide a comprehensive research report answering the user's question.

USER'S RESEARCH QUESTION: {request.research_question}

TOPIC: {request.topic}

COLLECTED DATA:
{collected_context}

Generate a detailed report that:
1. Directly answers the research question using the collected data
2. Provides specific recommendations based on the real data
3. Cites the data sources where applicable
4. Identifies any limitations or gaps in the available data
5. Provides actionable next steps

Format the report with clear sections and be specific - use actual numbers and data points from the collected information."""

        response = self._llm_assistant.client.generate(
            prompt,
            system_prompt="You are a research analyst. Synthesize the provided real-time data into actionable insights. Always cite specific data points from the collected information.",
        )

        return response.content

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

        if response.ai_direct_research:
            md += f"""## 🔬 AI Research Findings

{response.ai_direct_research}

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
