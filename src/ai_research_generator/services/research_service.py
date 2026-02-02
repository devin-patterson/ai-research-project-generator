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
    CollectedDataSummary,
    ExtractedEntities,
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
from ..tools.news_aggregator import NewsAggregatorTool, NewsAggregatorConfig
from ..tools.market_data import MarketDataTool, MarketDataConfig
from ..tools.international_economic import (
    IMFDataTool,
    WorldBankExtendedTool,
    InternationalEconomicConfig,
)
from ..tools.financial_markets import (
    FinnhubTool,
    FinancialMarketsConfig,
)
from ..tools.industry_data import (
    SECEdgarTool,
    IndustryDataConfig,
)
from ..tools.social_sentiment import (
    SocialSentimentTool,
    SocialSentimentConfig,
)

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
        self._news_aggregator: Optional[NewsAggregatorTool] = None
        self._market_data_tool: Optional[MarketDataTool] = None

        # New extended data tools
        self._imf_tool: Optional[IMFDataTool] = None
        self._world_bank_tool: Optional[WorldBankExtendedTool] = None
        self._finnhub_tool: Optional[FinnhubTool] = None
        self._sec_edgar_tool: Optional[SECEdgarTool] = None
        self._social_sentiment_tool: Optional[SocialSentimentTool] = None

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

        # Initialize new data aggregation tools
        self._news_aggregator = NewsAggregatorTool(NewsAggregatorConfig())
        self._market_data_tool = MarketDataTool(MarketDataConfig())

        # Initialize extended data tools (feature-flagged)
        self._imf_tool = IMFDataTool(tool_config, InternationalEconomicConfig())
        self._world_bank_tool = WorldBankExtendedTool(tool_config, InternationalEconomicConfig())
        self._finnhub_tool = FinnhubTool(tool_config, FinancialMarketsConfig())
        self._sec_edgar_tool = SECEdgarTool(tool_config, IndustryDataConfig())
        self._social_sentiment_tool = SocialSentimentTool(tool_config, SocialSentimentConfig())

        logger.info("Research toolkit initialized")
        logger.info(
            f"Economic data sources available: {self._economic_tool.get_available_sources()}"
        )
        logger.info(f"News sources available: {self._news_aggregator.get_available_sources()}")
        logger.info(
            f"Market data sources available: {self._market_data_tool.get_available_sources()}"
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
                collected_data_summary=ai_content.get("collected_data_summary"),
                extracted_entities=ai_content.get("extracted_entities"),
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
                    abstract = paper.get("abstract") or "No abstract available"
                    text = (
                        f"Academic Paper: {paper['title']}\n"
                        f"Authors: {authors_str}\n"
                        f"Year: {paper.get('year') or 'N/A'}\n"
                        f"Abstract: {abstract[:500]}\n"
                        f"DOI: {paper.get('doi') or 'N/A'}"
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

    async def _extract_topic_entities(self, request: ResearchRequest) -> ExtractedEntities:
        """
        Use LLM to extract structured entities from the research topic.

        This enables targeted data collection based on what's actually in the topic.
        """
        prompt = f"""Extract key entities from this research topic for data collection.

TOPIC: {request.topic}
RESEARCH QUESTION: {request.research_question}
DISCIPLINE: {request.discipline}
ADDITIONAL CONTEXT: {request.additional_context or "None provided"}

Return a JSON object with these fields (use empty arrays if none found):
{{
    "companies": ["company names and stock tickers mentioned"],
    "industries": ["industry sectors like healthcare, technology, finance"],
    "geographic_regions": ["countries, states, cities mentioned"],
    "time_periods": ["years, quarters, date ranges mentioned"],
    "key_metrics": ["KPIs, metrics, measurements to track"],
    "people": ["notable people mentioned"],
    "keywords": ["key search terms for finding relevant data"]
}}

Be thorough - extract ALL relevant entities for comprehensive data collection."""

        try:
            response = self._llm_assistant.client.generate(
                prompt,
                system_prompt="You are an entity extraction specialist. Return ONLY valid JSON, no other text.",
            )

            import json
            import re

            # Extract JSON from response
            content = response.content.strip()
            # Try to find JSON in the response
            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                entities_dict = json.loads(json_match.group())
            else:
                entities_dict = json.loads(content)

            return ExtractedEntities(
                companies=entities_dict.get("companies", []),
                industries=entities_dict.get("industries", []),
                geographic_regions=entities_dict.get("geographic_regions", []),
                time_periods=entities_dict.get("time_periods", []),
                key_metrics=entities_dict.get("key_metrics", []),
                people=entities_dict.get("people", []),
                keywords=entities_dict.get("keywords", []),
            )
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}, using fallback")
            # Fallback: basic keyword extraction
            topic_words = request.topic.split()
            return ExtractedEntities(
                companies=[w for w in topic_words if w[0].isupper() and len(w) > 2][:5],
                industries=[request.discipline] if request.discipline else [],
                geographic_regions=[],
                time_periods=[],
                key_metrics=[],
                people=[],
                keywords=topic_words[:10],
            )

    def _build_collected_data_summary(
        self, collected_data: dict, collection_start_time: datetime
    ) -> CollectedDataSummary:
        """Build a summary of what data was collected from which sources."""
        sources_queried = list(collected_data.keys())
        sources_successful = []
        sources_failed = []
        total_data_points = 0

        for source, data in collected_data.items():
            if data.get("error"):
                sources_failed.append(source)
            else:
                sources_successful.append(source)
                # Count data points based on source type
                if "indicators" in data:
                    total_data_points += len(data["indicators"])
                if "articles" in data:
                    total_data_points += len(data["articles"])
                if "events" in data:
                    total_data_points += len(data["events"])
                if "filings" in data:
                    total_data_points += len(data["filings"])
                if "quotes" in data:
                    total_data_points += len(data["quotes"])
                if "results" in data:
                    total_data_points += len(data["results"])
                if "posts" in data:
                    total_data_points += len(data["posts"])

        collection_duration = (datetime.now() - collection_start_time).total_seconds()

        return CollectedDataSummary(
            sources_queried=sources_queried,
            sources_successful=sources_successful,
            sources_failed=sources_failed,
            data_points_collected=total_data_points,
            collection_timestamp=datetime.now(),
            cache_hits=0,  # TODO: track cache hits
            collection_duration_seconds=collection_duration,
        )

    async def _enhance_with_llm(self, request: ResearchRequest, papers: list) -> Tuple[dict, str]:
        """
        Enhance project with LLM-generated content.

        This method now follows a data-driven approach:
        1. Extract entities from topic for targeted data collection
        2. Plan what data/tools are needed based on the prompt
        3. Execute tools to gather real data
        4. Synthesize the real data into the final report
        """
        content = {}
        model_used = None
        collected_data = {}
        collection_start_time = datetime.now()

        try:
            # Step 1: Extract entities from topic for targeted data collection
            logger.info("Extracting entities from research topic...")
            extracted_entities = await self._extract_topic_entities(request)
            content["extracted_entities"] = extracted_entities
            logger.info(
                f"Extracted entities: companies={extracted_entities.companies}, industries={extracted_entities.industries}, regions={extracted_entities.geographic_regions}"
            )

            # Step 2: Analyze topic to understand data needs
            result = self._llm_assistant.analyze_topic(request.topic, request.discipline)
            content["topic_analysis"] = result["analysis"]
            model_used = result.get("model")

            # Step 3: Plan and execute data collection based on topic
            logger.info("Planning data collection based on prompt analysis...")
            tool_plan = self._plan_tool_execution(request)

            # Step 4: Execute planned tools to gather REAL data
            if tool_plan.get("needs_economic_data") and self._economic_tool:
                logger.info("Executing economic data collection...")
                collected_data["economic"] = await self._collect_economic_data(request)

            if tool_plan.get("needs_current_events") and self._current_events_tool:
                logger.info("Executing current events research...")
                collected_data["current_events"] = await self._collect_current_events(request)

            if tool_plan.get("needs_web_search") and self._web_search_tool:
                logger.info("Executing web search...")
                collected_data["web_search"] = await self._collect_web_data(request)

            # Collect news articles from multiple sources
            if self._news_aggregator:
                logger.info("Collecting news articles...")
                collected_data["news"] = await self._collect_news_articles(request)

            # Collect market data (for finance/economics topics)
            if self._market_data_tool and request.discipline:
                discipline_lower = request.discipline.lower()
                if any(
                    t in discipline_lower for t in ["finance", "economics", "business", "market"]
                ):
                    logger.info("Collecting market data...")
                    collected_data["market"] = await self._collect_market_data(request)

            # Collect international economic data (IMF, World Bank)
            if self._imf_tool and self._imf_tool.is_available:
                logger.info("Collecting IMF international economic data...")
                collected_data["imf"] = await self._collect_imf_data(request)

            if self._world_bank_tool and self._world_bank_tool.is_available:
                logger.info("Collecting World Bank development indicators...")
                collected_data["world_bank"] = await self._collect_world_bank_data(request)

            # Collect financial market data (Finnhub - if API key available)
            if self._finnhub_tool and self._finnhub_tool.is_available:
                logger.info("Collecting Finnhub financial data...")
                collected_data["finnhub"] = await self._collect_finnhub_data(request)

            # Collect SEC filings for company research
            if self._sec_edgar_tool and self._sec_edgar_tool.is_available:
                # Only for business/finance topics or if company mentioned
                if request.discipline and any(
                    t in request.discipline.lower()
                    for t in ["business", "finance", "economics", "management"]
                ):
                    logger.info("Collecting SEC EDGAR filings...")
                    collected_data["sec_filings"] = await self._collect_sec_data(request)

            # Collect social sentiment (Reddit/Twitter - if API keys available)
            if self._social_sentiment_tool and self._social_sentiment_tool.is_available:
                logger.info("Collecting social media sentiment...")
                collected_data["social_sentiment"] = await self._collect_social_sentiment(request)

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

            # Build collected data summary
            content["collected_data_summary"] = self._build_collected_data_summary(
                collected_data, collection_start_time
            )

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

    async def _collect_news_articles(self, request: ResearchRequest) -> dict:
        """Collect news articles from multiple sources."""
        cache_key = f"news_{request.topic[:50]}"

        # Check cache first
        if self._data_cache:
            cached = self._data_cache.get(cache_key)
            if cached:
                logger.info("Using cached news data")
                return cached

        if not self._news_aggregator:
            return {"error": "News aggregator not initialized", "articles": []}

        try:
            # Determine category based on discipline
            category = None
            if request.discipline:
                discipline_lower = request.discipline.lower()
                if any(t in discipline_lower for t in ["finance", "economics", "business"]):
                    category = "business"
                elif any(t in discipline_lower for t in ["health", "medical", "healthcare"]):
                    category = "health"
                elif any(t in discipline_lower for t in ["tech", "computer", "software"]):
                    category = "technology"

            news_result = await self._news_aggregator.aggregate_news(
                query=request.topic,
                category=category,
                days_back=30,
                max_results=15,
            )

            result = {
                "query": news_result.query,
                "total_count": news_result.total_count,
                "sources_used": news_result.sources_used,
                "articles": [
                    {
                        "title": a.title,
                        "description": a.description,
                        "source": a.source,
                        "url": a.url,
                        "published_at": a.published_at,
                        "category": a.category,
                        "provider": a.provider,
                    }
                    for a in news_result.articles[:15]
                ],
            }

            # Cache the result
            if self._data_cache:
                self._data_cache.set(cache_key, result, data_type="current_events")
                logger.info("Cached news data")

            # Ingest into RAG
            if self._rag_service and self._rag_service.is_available:
                for article in result["articles"][:10]:
                    text = f"News Article: {article['title']}\nSource: {article['source']}\nDate: {article['published_at']}\nSummary: {article['description']}"
                    self._rag_service.ingest_text(
                        text,
                        metadata={
                            "type": "news_article",
                            "source": article["source"],
                            "provider": article["provider"],
                        },
                        collection="current_events",
                    )
                logger.info(f"Ingested {min(len(result['articles']), 10)} news articles into RAG")

            return result
        except Exception as e:
            logger.warning(f"News collection failed: {e}")
            return {"error": str(e), "articles": []}

    async def _collect_market_data(self, request: ResearchRequest) -> dict:
        """Collect market and financial data."""
        cache_key = "market_data_overview"

        # Check cache first
        if self._data_cache:
            cached = self._data_cache.get(cache_key)
            if cached:
                logger.info("Using cached market data")
                return cached

        if not self._market_data_tool:
            return {"error": "Market data tool not initialized", "quotes": [], "indicators": []}

        try:
            market_result = await self._market_data_tool.get_market_overview(
                include_indicators=True,
                country="US",
            )

            result = {
                "timestamp": market_result.timestamp,
                "sources_used": market_result.sources_used,
                "market_summary": market_result.market_summary,
                "quotes": [
                    {
                        "symbol": q.symbol,
                        "name": q.name,
                        "price": q.price,
                        "change_percent": q.change_percent,
                        "volume": q.volume,
                    }
                    for q in market_result.quotes
                ],
                "indicators": [
                    {
                        "name": i.name,
                        "value": i.value,
                        "year": i.year,
                        "country": i.country,
                    }
                    for i in market_result.indicators
                ],
            }

            # Cache the result
            if self._data_cache:
                self._data_cache.set(cache_key, result, data_type="economic_data")
                logger.info("Cached market data")

            # Ingest into RAG
            if self._rag_service and self._rag_service.is_available:
                # Ingest market quotes
                for quote in result["quotes"]:
                    text = f"Market Quote: {quote['symbol']} ({quote['name']})\nPrice: ${quote['price']:.2f}\nChange: {quote['change_percent']:.2f}%"
                    self._rag_service.ingest_text(
                        text,
                        metadata={"type": "market_quote", "symbol": quote["symbol"]},
                        collection="economic_data",
                    )
                # Ingest economic indicators
                for ind in result["indicators"]:
                    text = f"World Bank Indicator: {ind['name']}\nValue: {ind['value']}\nYear: {ind['year']}\nCountry: {ind['country']}"
                    self._rag_service.ingest_text(
                        text,
                        metadata={"type": "world_bank_indicator", "name": ind["name"]},
                        collection="economic_data",
                    )
                logger.info("Ingested market data into RAG")

            return result
        except Exception as e:
            logger.warning(f"Market data collection failed: {e}")
            return {"error": str(e), "quotes": [], "indicators": []}

    async def _collect_imf_data(self, request: ResearchRequest) -> dict:
        """Collect IMF international economic data."""
        if not self._imf_tool:
            return {"error": "IMF tool not initialized", "indicators": []}

        try:
            result = await self._imf_tool.execute(
                indicators=["NGDP_RPCH", "PCPIPCH", "LUR"],
                countries=["USA", "CHN", "DEU", "JPN", "GBR"],
                start_year=2022,
            )
            return {
                "source": "IMF",
                "indicators": [
                    {
                        "name": ind.name,
                        "country": ind.country,
                        "value": ind.value,
                        "year": ind.year,
                    }
                    for ind in result.indicators[:20]
                ],
            }
        except Exception as e:
            logger.warning(f"IMF data collection failed: {e}")
            return {"error": str(e), "indicators": []}

    async def _collect_world_bank_data(self, request: ResearchRequest) -> dict:
        """Collect World Bank development indicators."""
        if not self._world_bank_tool:
            return {"error": "World Bank tool not initialized", "indicators": []}

        try:
            result = await self._world_bank_tool.execute(
                category="economic",
                countries=["US", "CN", "DE", "JP", "GB"],
                start_year=2020,
            )
            return {
                "source": "World Bank",
                "indicators": [
                    {
                        "name": ind.name,
                        "country": ind.country,
                        "value": ind.value,
                        "year": ind.year,
                    }
                    for ind in result.indicators[:20]
                ],
            }
        except Exception as e:
            logger.warning(f"World Bank data collection failed: {e}")
            return {"error": str(e), "indicators": []}

    async def _collect_finnhub_data(self, request: ResearchRequest) -> dict:
        """Collect Finnhub financial market data."""
        if not self._finnhub_tool or not self._finnhub_tool.is_available:
            return {"error": "Finnhub not available", "quotes": [], "news": []}

        try:
            result = await self._finnhub_tool.execute(
                symbols=["AAPL", "MSFT", "GOOGL", "AMZN"],
                include_news=True,
                include_profiles=False,
                include_financials=True,
            )
            return {
                "source": "Finnhub",
                "quotes": [
                    {
                        "symbol": q.symbol,
                        "price": q.price,
                        "change_percent": q.change_percent,
                    }
                    for q in result.quotes
                ],
                "news": [{"headline": n.headline, "source": n.source} for n in result.news[:10]],
            }
        except Exception as e:
            logger.warning(f"Finnhub data collection failed: {e}")
            return {"error": str(e), "quotes": [], "news": []}

    async def _collect_sec_data(self, request: ResearchRequest) -> dict:
        """Collect SEC EDGAR filings."""
        if not self._sec_edgar_tool:
            return {"error": "SEC EDGAR tool not initialized", "filings": []}

        try:
            # Extract company names from topic
            topic_words = request.topic.split()
            company = None
            for word in topic_words:
                if word[0].isupper() and len(word) > 2:
                    company = word
                    break

            if not company:
                return {"filings": []}

            result = await self._sec_edgar_tool.execute(
                company=company,
                form_types=["10-K", "10-Q", "8-K"],
                limit=5,
            )
            return {
                "source": "SEC EDGAR",
                "filings": [
                    {
                        "company": f.company_name,
                        "form_type": f.form_type,
                        "filing_date": f.filing_date,
                        "url": f.url,
                    }
                    for f in result.filings
                ],
            }
        except Exception as e:
            logger.warning(f"SEC EDGAR data collection failed: {e}")
            return {"error": str(e), "filings": []}

    async def _collect_social_sentiment(self, request: ResearchRequest) -> dict:
        """Collect social media sentiment analysis."""
        if not self._social_sentiment_tool or not self._social_sentiment_tool.is_available:
            return {"error": "Social sentiment not available", "posts": []}

        try:
            result = await self._social_sentiment_tool.execute(
                query=request.topic,
                limit=20,
            )
            sentiment = result.sentiment
            return {
                "source": ", ".join(result.sources_used),
                "overall_sentiment": sentiment.overall_sentiment if sentiment else "unknown",
                "sentiment_score": sentiment.sentiment_score if sentiment else 0,
                "positive_count": sentiment.positive_count if sentiment else 0,
                "negative_count": sentiment.negative_count if sentiment else 0,
                "sample_posts": [
                    {"title": p.title or p.content[:100], "source": p.source}
                    for p in result.posts[:5]
                ],
            }
        except Exception as e:
            logger.warning(f"Social sentiment collection failed: {e}")
            return {"error": str(e), "posts": []}

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

        # Add news articles context
        if "news" in collected_data and collected_data["news"].get("articles"):
            news = collected_data["news"]
            context_parts.append(
                f"\n## Recent News ({news.get('total_count', 0)} articles from {', '.join(news.get('sources_used', []))})"
            )
            for article in news.get("articles", [])[:10]:
                context_parts.append(f"- [{article['source']}] {article['title']}")
                if article.get("description"):
                    context_parts.append(f"  Summary: {article['description'][:150]}...")

        # Add market data context
        if "market" in collected_data:
            market = collected_data["market"]
            if market.get("quotes"):
                context_parts.append("\n## Market Data (Real-Time)")
                summary = market.get("market_summary", {})
                if summary:
                    context_parts.append(f"Market Trend: {summary.get('market_trend', 'N/A')}")
                    context_parts.append(f"Volatility: {summary.get('volatility', 'N/A')}")
                context_parts.append("\nMajor Indices:")
                for q in market.get("quotes", []):
                    context_parts.append(
                        f"- {q['symbol']} ({q['name']}): ${q['price']:.2f} ({q['change_percent']:+.2f}%)"
                    )

            if market.get("indicators"):
                context_parts.append("\nWorld Bank Economic Indicators:")
                for ind in market.get("indicators", []):
                    context_parts.append(f"- {ind['name']}: {ind['value']:.2f} ({ind['year']})")

        # Add IMF data context
        if "imf" in collected_data and collected_data["imf"].get("indicators"):
            context_parts.append("\n## IMF International Economic Data")
            for ind in collected_data["imf"].get("indicators", [])[:10]:
                context_parts.append(
                    f"- {ind['country']} {ind['name']}: {ind['value']:.2f}% ({ind['year']})"
                )

        # Add World Bank data context
        if "world_bank" in collected_data and collected_data["world_bank"].get("indicators"):
            context_parts.append("\n## World Bank Development Indicators")
            for ind in collected_data["world_bank"].get("indicators", [])[:10]:
                context_parts.append(
                    f"- {ind['country']} {ind['name']}: {ind['value']:.2f} ({ind['year']})"
                )

        # Add Finnhub financial data context
        if "finnhub" in collected_data:
            finnhub = collected_data["finnhub"]
            if finnhub.get("quotes"):
                context_parts.append("\n## Finnhub Financial Market Data")
                for q in finnhub.get("quotes", []):
                    context_parts.append(
                        f"- {q['symbol']}: ${q['price']:.2f} ({q['change_percent']:+.2f}%)"
                    )
            if finnhub.get("news"):
                context_parts.append("\nMarket News:")
                for n in finnhub.get("news", [])[:5]:
                    context_parts.append(f"- [{n['source']}] {n['headline']}")

        # Add SEC filings context
        if "sec_filings" in collected_data and collected_data["sec_filings"].get("filings"):
            context_parts.append("\n## SEC EDGAR Filings")
            for f in collected_data["sec_filings"].get("filings", []):
                context_parts.append(f"- {f['company']} {f['form_type']} ({f['filing_date']})")

        # Add social sentiment context
        if "social_sentiment" in collected_data:
            sentiment = collected_data["social_sentiment"]
            if sentiment.get("overall_sentiment"):
                context_parts.append("\n## Social Media Sentiment Analysis")
                context_parts.append(f"Overall Sentiment: {sentiment['overall_sentiment']}")
                context_parts.append(f"Sentiment Score: {sentiment.get('sentiment_score', 0):.2f}")
                context_parts.append(
                    f"Positive: {sentiment.get('positive_count', 0)}, "
                    f"Negative: {sentiment.get('negative_count', 0)}"
                )
                if sentiment.get("sample_posts"):
                    context_parts.append("Sample Posts:")
                    for p in sentiment.get("sample_posts", [])[:3]:
                        context_parts.append(f"- [{p['source']}] {p['title'][:80]}...")

        # Add academic papers context
        if papers:
            context_parts.append(f"\n## Academic Research ({len(papers)} papers found)")
            for p in papers[:5]:
                context_parts.append(f"- {p['title']} ({p.get('year', 'N/A')})")

        collected_context = "\n".join(context_parts)

        # Build list of data sources used for citation
        sources_used = [k for k, v in collected_data.items() if v and not v.get("error")]

        # Get discipline-specific system prompt and instructions
        system_prompt, discipline_instructions = self._get_discipline_specific_prompts(
            request.discipline
        )

        # Generate synthesis using LLM with real data
        prompt = f"""Based on the following REAL DATA collected from multiple sources, provide a comprehensive research report answering the user's question.

USER'S RESEARCH QUESTION: {request.research_question}

TOPIC: {request.topic}

DISCIPLINE: {request.discipline}

DATA SOURCES USED: {", ".join(sources_used)}

COLLECTED DATA:
{collected_context}

{discipline_instructions}

IMPORTANT CITATION REQUIREMENTS:
- Always cite specific data sources when making claims (e.g., "According to IMF data...", "SEC filings show...")
- Include specific numbers, percentages, and dates from the collected data
- Clearly distinguish between data-backed findings and analytical conclusions
- Note any limitations or gaps in the available data

Format the report with clear sections and be specific - use actual numbers and data points from the collected information."""

        response = self._llm_assistant.client.generate(
            prompt,
            system_prompt=system_prompt,
        )

        return response.content

    def _get_discipline_specific_prompts(self, discipline: str) -> tuple[str, str]:
        """
        Get discipline-specific system prompt and instructions for report synthesis.

        Returns:
            Tuple of (system_prompt, discipline_instructions)
        """
        discipline_lower = discipline.lower() if discipline else "general"

        # Discipline-specific system prompts
        system_prompts = {
            "finance": """You are a CFA-certified senior financial analyst with expertise in investment research, 
portfolio management, and market analysis. You provide rigorous, data-driven analysis with specific 
attention to risk factors, valuation metrics, and market conditions. Always cite specific data points 
and provide actionable investment insights with clear rationale.""",
            "healthcare": """You are a senior healthcare industry consultant with expertise in revenue cycle 
management, healthcare operations, regulatory compliance, and health IT. You understand CMS regulations, 
payer dynamics, and operational best practices. Provide actionable recommendations for healthcare executives 
with specific metrics, benchmarks, and implementation timelines.""",
            "economics": """You are a senior economist with expertise in macroeconomic analysis, monetary policy, 
and economic forecasting. You analyze economic indicators, policy impacts, and market dynamics with 
rigorous methodology. Provide data-driven insights with clear causal reasoning and uncertainty quantification.""",
            "technology": """You are a senior technology analyst with expertise in emerging technologies, 
market dynamics, and competitive analysis. You understand technical architectures, adoption curves, 
and business model implications. Provide strategic insights with specific market data and trend analysis.""",
            "business": """You are a senior management consultant with expertise in corporate strategy, 
operations, and organizational effectiveness. You provide actionable recommendations backed by 
industry benchmarks, competitive analysis, and best practices. Focus on implementation feasibility 
and measurable outcomes.""",
            "general": """You are a research analyst specializing in synthesizing data from multiple sources 
into actionable insights. You provide comprehensive, balanced analysis with clear citations and 
evidence-based recommendations. Always distinguish between data-backed findings and analytical conclusions.""",
        }

        # Discipline-specific report instructions
        discipline_instructions = {
            "finance": """Generate a detailed financial analysis report with these sections:
1. **Executive Summary** - Key findings and investment thesis
2. **Market Analysis** - Current market conditions and trends
3. **Fundamental Analysis** - Key financial metrics and ratios
4. **Risk Assessment** - Market, regulatory, and company-specific risks
5. **Valuation Analysis** - Fair value estimates with methodology
6. **Recommendations** - Specific actionable recommendations with rationale
7. **Limitations** - Data gaps and analytical caveats""",
            "healthcare": """Generate a detailed healthcare industry report with these sections:
1. **Executive Summary** - Key findings and strategic recommendations
2. **Regulatory Landscape** - Current and upcoming regulatory changes
3. **Market Analysis** - Industry trends, payer dynamics, reimbursement changes
4. **Operational Analysis** - Performance metrics, benchmarks, best practices
5. **Staffing & Resources** - FTE recommendations, skill requirements, training needs
6. **Technology & Innovation** - Relevant technology trends and adoption recommendations
7. **Implementation Roadmap** - Prioritized action items with timelines
8. **Risk Mitigation** - Key risks and mitigation strategies
9. **Limitations** - Data gaps and areas requiring further research""",
            "economics": """Generate a detailed economic analysis report with these sections:
1. **Executive Summary** - Key economic findings and outlook
2. **Macroeconomic Indicators** - GDP, inflation, employment analysis
3. **Monetary & Fiscal Policy** - Policy impacts and expectations
4. **Sector Analysis** - Industry-specific economic trends
5. **Forecasts** - Economic projections with confidence intervals
6. **Risk Factors** - Economic risks and scenarios
7. **Policy Implications** - Recommendations for stakeholders
8. **Limitations** - Data limitations and forecast uncertainty""",
            "technology": """Generate a detailed technology analysis report with these sections:
1. **Executive Summary** - Key findings and strategic implications
2. **Technology Landscape** - Current state and emerging trends
3. **Market Analysis** - Market size, growth, competitive dynamics
4. **Adoption Analysis** - Adoption curves, barriers, enablers
5. **Competitive Analysis** - Key players and positioning
6. **Strategic Recommendations** - Actionable technology strategy
7. **Implementation Considerations** - Technical and organizational factors
8. **Limitations** - Data gaps and analytical caveats""",
            "business": """Generate a detailed business analysis report with these sections:
1. **Executive Summary** - Key findings and strategic recommendations
2. **Industry Analysis** - Market dynamics and competitive landscape
3. **Operational Analysis** - Performance metrics and benchmarks
4. **Strategic Options** - Alternative approaches with pros/cons
5. **Financial Impact** - ROI analysis and resource requirements
6. **Implementation Plan** - Prioritized action items with timelines
7. **Risk Assessment** - Key risks and mitigation strategies
8. **Limitations** - Data gaps and areas for further analysis""",
            "general": """Generate a detailed research report with these sections:
1. **Executive Summary** - Key findings and recommendations
2. **Background & Context** - Relevant context and scope
3. **Data Analysis** - Analysis of collected data with citations
4. **Key Findings** - Main insights from the research
5. **Recommendations** - Actionable recommendations with rationale
6. **Limitations** - Data gaps and analytical caveats
7. **Next Steps** - Suggested follow-up actions""",
        }

        # Match discipline to closest category
        matched_discipline = "general"
        for key in system_prompts.keys():
            if key in discipline_lower or discipline_lower in key:
                matched_discipline = key
                break

        # Special handling for common variations
        if any(
            term in discipline_lower
            for term in ["revenue", "medical", "hospital", "clinical", "pharma"]
        ):
            matched_discipline = "healthcare"
        elif any(
            term in discipline_lower
            for term in ["invest", "stock", "market", "trading", "portfolio"]
        ):
            matched_discipline = "finance"
        elif any(
            term in discipline_lower for term in ["software", "ai", "data", "digital", "cyber"]
        ):
            matched_discipline = "technology"
        elif any(
            term in discipline_lower
            for term in ["management", "strategy", "operations", "consulting"]
        ):
            matched_discipline = "business"

        return system_prompts[matched_discipline], discipline_instructions[matched_discipline]

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

        # Add data collection summary if available
        if response.collected_data_summary:
            summary = response.collected_data_summary
            md += f"""## 📊 Data Collection Summary

**Sources Queried**: {len(summary.sources_queried)}  
**Sources Successful**: {len(summary.sources_successful)} ({", ".join(summary.sources_successful) if summary.sources_successful else "None"})  
**Sources Failed**: {len(summary.sources_failed)} ({", ".join(summary.sources_failed) if summary.sources_failed else "None"})  
**Data Points Collected**: {summary.data_points_collected}  
**Collection Time**: {summary.collection_duration_seconds:.2f}s  

---

"""

        # Add extracted entities if available
        if response.extracted_entities:
            entities = response.extracted_entities
            md += """## 🔍 Extracted Entities

"""
            if entities.companies:
                md += f"**Companies**: {', '.join(entities.companies)}\n"
            if entities.industries:
                md += f"**Industries**: {', '.join(entities.industries)}\n"
            if entities.geographic_regions:
                md += f"**Geographic Regions**: {', '.join(entities.geographic_regions)}\n"
            if entities.time_periods:
                md += f"**Time Periods**: {', '.join(entities.time_periods)}\n"
            if entities.key_metrics:
                md += f"**Key Metrics**: {', '.join(entities.key_metrics)}\n"
            if entities.keywords:
                md += f"**Keywords**: {', '.join(entities.keywords[:10])}\n"
            md += "\n---\n\n"

        md += f"""## ✅ Validation Report

**Quality Score**: {response.validation_report.overall_score:.2f}/1.00

### Recommendations

"""
        for rec in response.validation_report.recommendations[:5]:
            md += f"- {rec}\n"

        md += "\n---\n\n*Generated by AI Research Project Generator API*\n"

        return md
