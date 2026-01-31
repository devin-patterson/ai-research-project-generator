"""
LangGraph-based research workflow orchestration.

This module implements a graph-based workflow for research project generation
using LangGraph's StateGraph for orchestration and checkpointing.
"""

from dataclasses import dataclass
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
from operator import add

from loguru import logger
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy

from ..core.retry import LLM_RETRY_CONFIG, ACADEMIC_SEARCH_RETRY_CONFIG


class ResearchState(TypedDict, total=False):
    """State schema for the research workflow graph.

    This TypedDict defines all the data that flows through the workflow,
    with each node reading from and writing to this shared state.
    """

    # Input
    topic: str
    research_question: str
    research_type: str
    discipline: str
    academic_level: str
    paper_limit: int
    year_start: Optional[int]
    year_end: Optional[int]

    # Configuration
    use_llm: bool
    search_papers: bool
    llm_model: str

    # Workflow state
    request_id: str
    current_step: str
    errors: Annotated[list[str], add]

    # Intermediate results
    topic_analysis: dict[str, Any]
    key_concepts: list[str]
    search_queries: list[str]
    papers: list[dict[str, Any]]
    methodology_recommendations: list[str]

    # Output
    research_context: dict[str, Any]
    subject_analysis: dict[str, Any]
    validation_report: dict[str, Any]
    quality_score: float
    final_output: dict[str, Any]


@dataclass
class ResearchWorkflow:
    """
    Orchestrates the research generation workflow using LangGraph.

    The workflow consists of the following nodes:
    1. analyze_topic - Analyze the research topic and extract key concepts
    2. generate_queries - Generate search queries from key concepts
    3. search_papers - Search academic databases for relevant papers
    4. analyze_papers - Analyze and synthesize paper findings
    5. recommend_methodology - Recommend research methodology
    6. validate_output - Validate the generated research project
    7. generate_report - Generate the final research report

    The graph supports:
    - Checkpointing for pause/resume
    - Retry policies for transient failures
    - Conditional routing based on configuration
    """

    llm_client: Any = None
    search_client: Any = None
    checkpointer: Optional[MemorySaver] = None

    def __post_init__(self):
        """Initialize the workflow graph after dataclass initialization."""
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build and compile the LangGraph workflow."""
        workflow = StateGraph(ResearchState)

        # Add nodes with retry policies
        workflow.add_node(
            "analyze_topic",
            self._analyze_topic,
            retry=RetryPolicy(max_attempts=LLM_RETRY_CONFIG.max_retries),
        )
        workflow.add_node(
            "generate_queries",
            self._generate_queries,
            retry=RetryPolicy(max_attempts=LLM_RETRY_CONFIG.max_retries),
        )
        workflow.add_node(
            "search_papers",
            self._search_papers,
            retry=RetryPolicy(max_attempts=ACADEMIC_SEARCH_RETRY_CONFIG.max_retries),
        )
        workflow.add_node(
            "analyze_papers",
            self._analyze_papers,
            retry=RetryPolicy(max_attempts=LLM_RETRY_CONFIG.max_retries),
        )
        workflow.add_node(
            "recommend_methodology",
            self._recommend_methodology,
            retry=RetryPolicy(max_attempts=LLM_RETRY_CONFIG.max_retries),
        )
        workflow.add_node("validate_output", self._validate_output)
        workflow.add_node("generate_report", self._generate_report)

        # Define edges (workflow flow)
        workflow.add_edge(START, "analyze_topic")
        workflow.add_edge("analyze_topic", "generate_queries")

        # Conditional edge: skip paper search if disabled
        workflow.add_conditional_edges(
            "generate_queries",
            self._should_search_papers,
            {True: "search_papers", False: "recommend_methodology"},
        )

        workflow.add_edge("search_papers", "analyze_papers")
        workflow.add_edge("analyze_papers", "recommend_methodology")
        workflow.add_edge("recommend_methodology", "validate_output")
        workflow.add_edge("validate_output", "generate_report")
        workflow.add_edge("generate_report", END)

        # Compile with checkpointer for persistence
        if self.checkpointer is None:
            self.checkpointer = MemorySaver()

        return workflow.compile(checkpointer=self.checkpointer)

    def _should_search_papers(self, state: ResearchState) -> bool:
        """Conditional routing: determine if paper search should be performed."""
        return state.get("search_papers", True)

    async def _analyze_topic(self, state: ResearchState) -> dict[str, Any]:
        """
        Node 1: Analyze the research topic.

        Extracts key concepts, identifies research gaps, and determines
        the scope of the research project.
        """
        logger.info(f"Analyzing topic: {state['topic']}")

        topic = state["topic"]
        research_question = state["research_question"]
        use_llm = state.get("use_llm", True)

        if use_llm and self.llm_client:
            # Use LLM for topic analysis
            analysis = await self._llm_analyze_topic(topic, research_question)
        else:
            # Fallback to rule-based analysis
            analysis = self._rule_based_topic_analysis(topic, research_question)

        return {
            "current_step": "analyze_topic",
            "topic_analysis": analysis,
            "key_concepts": analysis.get("key_concepts", []),
        }

    async def _generate_queries(self, state: ResearchState) -> dict[str, Any]:
        """
        Node 2: Generate search queries from key concepts.

        Creates optimized search queries for academic databases based on
        the extracted key concepts and research question.
        """
        logger.info("Generating search queries")

        key_concepts = state.get("key_concepts", [])
        topic = state["topic"]
        research_question = state["research_question"]

        # Generate search queries
        queries = []

        # Primary query from topic
        queries.append(topic)

        # Query from research question
        queries.append(research_question)

        # Queries from key concepts
        for concept in key_concepts[:5]:  # Limit to top 5 concepts
            queries.append(f"{topic} {concept}")

        return {
            "current_step": "generate_queries",
            "search_queries": queries,
        }

    async def _search_papers(self, state: ResearchState) -> dict[str, Any]:
        """
        Node 3: Search academic databases for relevant papers.

        Searches multiple academic APIs (Semantic Scholar, OpenAlex, CrossRef, arXiv)
        and aggregates results.
        """
        logger.info("Searching for academic papers")

        queries = state.get("search_queries", [state["topic"]])
        paper_limit = state.get("paper_limit", 20)
        year_start = state.get("year_start")
        year_end = state.get("year_end")

        papers = []
        errors = []

        if self.search_client:
            for query in queries[:3]:  # Limit queries to avoid rate limits
                try:
                    results = await self.search_client.search(
                        query=query,
                        limit=paper_limit // len(queries[:3]),
                        year_start=year_start,
                        year_end=year_end,
                    )
                    papers.extend(results)
                except Exception as e:
                    logger.warning(f"Search failed for query '{query}': {e}")
                    errors.append(f"Search error: {str(e)}")

        # Deduplicate papers by DOI or title
        seen = set()
        unique_papers = []
        for paper in papers:
            key = paper.get("doi") or paper.get("title", "")
            if key and key not in seen:
                seen.add(key)
                unique_papers.append(paper)

        return {
            "current_step": "search_papers",
            "papers": unique_papers[:paper_limit],
            "errors": errors,
        }

    async def _analyze_papers(self, state: ResearchState) -> dict[str, Any]:
        """
        Node 4: Analyze and synthesize paper findings.

        Extracts themes, identifies gaps, and synthesizes findings
        from the collected papers.
        """
        logger.info("Analyzing papers")

        papers = state.get("papers", [])
        use_llm = state.get("use_llm", True)

        if not papers:
            return {
                "current_step": "analyze_papers",
                "research_context": {
                    "papers_analyzed": 0,
                    "themes": [],
                    "gaps": [],
                    "synthesis": "No papers found for analysis.",
                },
            }

        if use_llm and self.llm_client:
            context = await self._llm_analyze_papers(papers)
        else:
            context = self._rule_based_paper_analysis(papers)

        return {
            "current_step": "analyze_papers",
            "research_context": context,
        }

    async def _recommend_methodology(self, state: ResearchState) -> dict[str, Any]:
        """
        Node 5: Recommend research methodology.

        Based on the research type, discipline, and findings, recommends
        appropriate research methodologies.
        """
        logger.info("Recommending methodology")

        research_type = state.get("research_type", "systematic_review")
        discipline = state.get("discipline", "general")
        academic_level = state.get("academic_level", "graduate")

        # Methodology recommendations based on research type
        methodology_map = {
            "systematic_review": [
                "PRISMA guidelines for systematic reviews",
                "Comprehensive database search strategy",
                "Quality assessment using validated tools",
                "Narrative synthesis or meta-analysis",
            ],
            "meta_analysis": [
                "Statistical pooling of effect sizes",
                "Heterogeneity assessment (I² statistic)",
                "Publication bias analysis (funnel plots)",
                "Sensitivity and subgroup analyses",
            ],
            "literature_review": [
                "Thematic analysis of literature",
                "Chronological or conceptual organization",
                "Critical evaluation of sources",
                "Identification of research gaps",
            ],
            "empirical_study": [
                "Hypothesis formulation",
                "Research design selection",
                "Data collection methodology",
                "Statistical analysis plan",
            ],
            "case_study": [
                "Case selection criteria",
                "Multiple data sources triangulation",
                "Pattern matching analysis",
                "Cross-case synthesis",
            ],
            "theoretical_framework": [
                "Concept mapping",
                "Theory building methodology",
                "Logical argumentation structure",
                "Framework validation approach",
            ],
        }

        recommendations = methodology_map.get(research_type, methodology_map["literature_review"])

        return {
            "current_step": "recommend_methodology",
            "methodology_recommendations": recommendations,
            "subject_analysis": {
                "research_type": research_type,
                "discipline": discipline,
                "academic_level": academic_level,
                "methodology": recommendations,
            },
        }

    async def _validate_output(self, state: ResearchState) -> dict[str, Any]:
        """
        Node 6: Validate the generated research project.

        Checks completeness, consistency, and quality of the generated
        research project components.
        """
        logger.info("Validating output")

        issues = []
        score = 100.0

        # Check topic analysis
        if not state.get("topic_analysis"):
            issues.append("Missing topic analysis")
            score -= 20

        # Check key concepts
        key_concepts = state.get("key_concepts", [])
        if len(key_concepts) < 3:
            issues.append("Insufficient key concepts identified")
            score -= 10

        # Check papers (if search was enabled)
        if state.get("search_papers", True):
            papers = state.get("papers", [])
            if len(papers) < 5:
                issues.append("Limited academic sources found")
                score -= 15

        # Check methodology
        if not state.get("methodology_recommendations"):
            issues.append("Missing methodology recommendations")
            score -= 15

        return {
            "current_step": "validate_output",
            "validation_report": {
                "issues": issues,
                "passed": len(issues) == 0,
                "issue_count": len(issues),
            },
            "quality_score": max(0, score),
        }

    async def _generate_report(self, state: ResearchState) -> dict[str, Any]:
        """
        Node 7: Generate the final research report.

        Compiles all components into a comprehensive research project report.
        """
        logger.info("Generating final report")

        final_output = {
            "request_id": state.get("request_id", ""),
            "topic": state["topic"],
            "research_question": state["research_question"],
            "research_type": state.get("research_type", "systematic_review"),
            "discipline": state.get("discipline", "general"),
            "academic_level": state.get("academic_level", "graduate"),
            "topic_analysis": state.get("topic_analysis", {}),
            "key_concepts": state.get("key_concepts", []),
            "papers": state.get("papers", []),
            "research_context": state.get("research_context", {}),
            "subject_analysis": state.get("subject_analysis", {}),
            "methodology_recommendations": state.get("methodology_recommendations", []),
            "validation_report": state.get("validation_report", {}),
            "quality_score": state.get("quality_score", 0),
        }

        return {
            "current_step": "generate_report",
            "final_output": final_output,
        }

    # Helper methods for LLM and rule-based analysis

    async def _llm_analyze_topic(self, topic: str, research_question: str) -> dict[str, Any]:
        """Use LLM to analyze the research topic."""
        # This will be implemented with PydanticAI
        # For now, fall back to rule-based
        return self._rule_based_topic_analysis(topic, research_question)

    def _rule_based_topic_analysis(self, topic: str, research_question: str) -> dict[str, Any]:
        """Rule-based topic analysis fallback."""
        # Extract key concepts from topic words
        words = topic.lower().split()
        stop_words = {"the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "are"}
        key_concepts = [w for w in words if w not in stop_words and len(w) > 3]

        return {
            "topic": topic,
            "research_question": research_question,
            "key_concepts": key_concepts[:10],
            "scope": "comprehensive",
            "complexity": "moderate",
        }

    async def _llm_analyze_papers(self, papers: list[dict]) -> dict[str, Any]:
        """Use LLM to analyze papers."""
        # This will be implemented with PydanticAI
        return self._rule_based_paper_analysis(papers)

    def _rule_based_paper_analysis(self, papers: list[dict]) -> dict[str, Any]:
        """Rule-based paper analysis fallback."""
        # Extract basic statistics
        years = [p.get("year") for p in papers if p.get("year")]

        return {
            "papers_analyzed": len(papers),
            "year_range": f"{min(years) if years else 'N/A'}-{max(years) if years else 'N/A'}",
            "themes": [],
            "gaps": [],
            "synthesis": f"Analyzed {len(papers)} papers on the research topic.",
        }

    async def run(
        self, topic: str, research_question: str, request_id: str = "", **kwargs
    ) -> dict[str, Any]:
        """
        Execute the research workflow.

        Args:
            topic: The research topic
            research_question: The main research question
            request_id: Unique identifier for this request
            **kwargs: Additional configuration options

        Returns:
            The final research project output
        """
        initial_state: ResearchState = {
            "topic": topic,
            "research_question": research_question,
            "request_id": request_id,
            "research_type": kwargs.get("research_type", "systematic_review"),
            "discipline": kwargs.get("discipline", "general"),
            "academic_level": kwargs.get("academic_level", "graduate"),
            "paper_limit": kwargs.get("paper_limit", 20),
            "year_start": kwargs.get("year_start"),
            "year_end": kwargs.get("year_end"),
            "use_llm": kwargs.get("use_llm", True),
            "search_papers": kwargs.get("search_papers", True),
            "llm_model": kwargs.get("llm_model", "llama3.1:8b"),
            "errors": [],
        }

        config = {"configurable": {"thread_id": request_id or "default"}}

        # Execute the workflow
        result = await self.graph.ainvoke(initial_state, config)

        return result.get("final_output", result)


def create_research_graph(
    llm_client: Any = None,
    search_client: Any = None,
) -> ResearchWorkflow:
    """
    Factory function to create a configured research workflow.

    Args:
        llm_client: Optional LLM client for AI-powered analysis
        search_client: Optional academic search client

    Returns:
        Configured ResearchWorkflow instance
    """
    return ResearchWorkflow(
        llm_client=llm_client,
        search_client=search_client,
    )
