"""
Tests for LangGraph research workflow.

This module tests the graph-based workflow orchestration including:
1. Individual node execution
2. Workflow state transitions
3. Conditional routing
4. Checkpointing and recovery
"""

import pytest
from unittest.mock import AsyncMock

from src.ai_research_generator.workflows.research_workflow import (
    ResearchWorkflow,
    ResearchState,
    create_research_graph,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = AsyncMock()
    client.generate = AsyncMock(return_value={"response": "Test response"})
    return client


@pytest.fixture
def mock_search_client():
    """Create a mock academic search client."""
    client = AsyncMock()
    client.search = AsyncMock(
        return_value=[
            {
                "title": "Test Paper 1",
                "abstract": "Abstract for test paper 1",
                "year": 2024,
                "doi": "10.1234/test1",
            },
            {
                "title": "Test Paper 2",
                "abstract": "Abstract for test paper 2",
                "year": 2023,
                "doi": "10.1234/test2",
            },
        ]
    )
    return client


@pytest.fixture
def workflow(mock_llm_client, mock_search_client):
    """Create a research workflow with mocked clients."""
    return ResearchWorkflow(
        llm_client=mock_llm_client,
        search_client=mock_search_client,
    )


@pytest.fixture
def sample_initial_state() -> ResearchState:
    """Create a sample initial state for testing."""
    return {
        "topic": "Impact of artificial intelligence on education",
        "research_question": "How does AI affect learning outcomes?",
        "research_type": "systematic_review",
        "discipline": "education",
        "academic_level": "graduate",
        "paper_limit": 20,
        "year_start": 2020,
        "year_end": 2024,
        "use_llm": False,  # Disable LLM for faster tests
        "search_papers": True,
        "llm_model": "llama3.1:8b",
        "request_id": "test-123",
        "errors": [],
    }


# =============================================================================
# Workflow Creation Tests
# =============================================================================


class TestWorkflowCreation:
    """Tests for workflow creation and configuration."""

    def test_create_workflow_with_defaults(self):
        """Test creating a workflow with default configuration."""
        workflow = create_research_graph()

        assert workflow is not None
        assert isinstance(workflow, ResearchWorkflow)
        assert workflow.checkpointer is not None

    def test_create_workflow_with_clients(self, mock_llm_client, mock_search_client):
        """Test creating a workflow with custom clients."""
        workflow = create_research_graph(
            llm_client=mock_llm_client,
            search_client=mock_search_client,
        )

        assert workflow.llm_client == mock_llm_client
        assert workflow.search_client == mock_search_client

    def test_workflow_graph_is_compiled(self, workflow):
        """Test that the workflow graph is properly compiled."""
        assert workflow.graph is not None
        # LangGraph compiled graphs have an invoke method
        assert hasattr(workflow.graph, "ainvoke")


# =============================================================================
# Individual Node Tests
# =============================================================================


class TestWorkflowNodes:
    """Tests for individual workflow nodes."""

    @pytest.mark.asyncio
    async def test_analyze_topic_node(self, workflow, sample_initial_state):
        """Test the topic analysis node."""
        result = await workflow._analyze_topic(sample_initial_state)

        assert "topic_analysis" in result
        assert "key_concepts" in result
        assert result["current_step"] == "analyze_topic"
        assert len(result["key_concepts"]) > 0

    @pytest.mark.asyncio
    async def test_generate_queries_node(self, workflow, sample_initial_state):
        """Test the query generation node."""
        # Add key concepts from previous step
        state = {
            **sample_initial_state,
            "key_concepts": ["artificial intelligence", "education", "learning"],
        }

        result = await workflow._generate_queries(state)

        assert "search_queries" in result
        assert result["current_step"] == "generate_queries"
        assert len(result["search_queries"]) > 0

    @pytest.mark.asyncio
    async def test_search_papers_node(self, workflow, sample_initial_state):
        """Test the paper search node."""
        state = {
            **sample_initial_state,
            "search_queries": ["AI in education", "machine learning education"],
        }

        result = await workflow._search_papers(state)

        assert "papers" in result
        assert result["current_step"] == "search_papers"
        # Mock client returns 2 papers
        assert len(result["papers"]) > 0

    @pytest.mark.asyncio
    async def test_search_papers_deduplicates(self, workflow, sample_initial_state):
        """Test that paper search deduplicates results."""
        # Configure mock to return duplicate papers
        workflow.search_client.search = AsyncMock(
            return_value=[
                {"title": "Same Paper", "doi": "10.1234/same", "year": 2024},
                {"title": "Same Paper", "doi": "10.1234/same", "year": 2024},
                {"title": "Different Paper", "doi": "10.1234/diff", "year": 2023},
            ]
        )

        state = {
            **sample_initial_state,
            "search_queries": ["test query"],
        }

        result = await workflow._search_papers(state)

        # Should have deduplicated to 2 unique papers
        assert len(result["papers"]) == 2

    @pytest.mark.asyncio
    async def test_analyze_papers_node_with_papers(self, workflow, sample_initial_state):
        """Test paper analysis node with papers."""
        state = {
            **sample_initial_state,
            "papers": [
                {"title": "Paper 1", "abstract": "Abstract 1", "year": 2024},
                {"title": "Paper 2", "abstract": "Abstract 2", "year": 2023},
            ],
        }

        result = await workflow._analyze_papers(state)

        assert "research_context" in result
        assert result["current_step"] == "analyze_papers"
        assert result["research_context"]["papers_analyzed"] == 2

    @pytest.mark.asyncio
    async def test_analyze_papers_node_without_papers(self, workflow, sample_initial_state):
        """Test paper analysis node without papers."""
        state = {
            **sample_initial_state,
            "papers": [],
        }

        result = await workflow._analyze_papers(state)

        assert result["research_context"]["papers_analyzed"] == 0
        assert "No papers found" in result["research_context"]["synthesis"]

    @pytest.mark.asyncio
    async def test_recommend_methodology_node(self, workflow, sample_initial_state):
        """Test methodology recommendation node."""
        result = await workflow._recommend_methodology(sample_initial_state)

        assert "methodology_recommendations" in result
        assert "subject_analysis" in result
        assert result["current_step"] == "recommend_methodology"
        assert len(result["methodology_recommendations"]) > 0

    @pytest.mark.asyncio
    async def test_recommend_methodology_for_different_types(self, workflow):
        """Test methodology recommendations vary by research type."""
        research_types = [
            "systematic_review",
            "meta_analysis",
            "case_study",
            "empirical_study",
        ]

        results = []
        for rt in research_types:
            state = {
                "research_type": rt,
                "discipline": "general",
                "academic_level": "graduate",
            }
            result = await workflow._recommend_methodology(state)
            results.append(result["methodology_recommendations"])

        # Different research types should have different recommendations
        # (at least some should differ)
        unique_recommendations = set(tuple(r) for r in results)
        assert len(unique_recommendations) > 1

    @pytest.mark.asyncio
    async def test_validate_output_node_passes(self, workflow):
        """Test validation node with complete state."""
        state = {
            "topic_analysis": {"topic": "Test", "key_concepts": ["a", "b", "c"]},
            "key_concepts": ["concept1", "concept2", "concept3"],
            "papers": [{"title": f"Paper {i}"} for i in range(10)],
            "methodology_recommendations": ["Method 1", "Method 2"],
            "search_papers": True,
        }

        result = await workflow._validate_output(state)

        assert result["validation_report"]["passed"] is True
        assert result["quality_score"] == 100.0

    @pytest.mark.asyncio
    async def test_validate_output_node_fails(self, workflow):
        """Test validation node with incomplete state."""
        state = {
            "topic_analysis": None,  # Missing
            "key_concepts": ["a"],  # Too few
            "papers": [],  # No papers
            "methodology_recommendations": None,  # Missing
            "search_papers": True,
        }

        result = await workflow._validate_output(state)

        assert result["validation_report"]["passed"] is False
        assert result["quality_score"] < 100.0
        assert len(result["validation_report"]["issues"]) > 0

    @pytest.mark.asyncio
    async def test_generate_report_node(self, workflow, sample_initial_state):
        """Test report generation node."""
        state = {
            **sample_initial_state,
            "topic_analysis": {"topic": "Test"},
            "key_concepts": ["concept1"],
            "papers": [{"title": "Paper 1"}],
            "research_context": {"synthesis": "Test synthesis"},
            "subject_analysis": {"methodology": ["Method 1"]},
            "methodology_recommendations": ["Method 1"],
            "validation_report": {"passed": True},
            "quality_score": 85.0,
        }

        result = await workflow._generate_report(state)

        assert "final_output" in result
        assert result["current_step"] == "generate_report"
        assert result["final_output"]["topic"] == sample_initial_state["topic"]
        assert result["final_output"]["quality_score"] == 85.0


# =============================================================================
# Conditional Routing Tests
# =============================================================================


class TestConditionalRouting:
    """Tests for conditional routing in the workflow."""

    def test_should_search_papers_true(self, workflow):
        """Test routing when paper search is enabled."""
        state = {"search_papers": True}
        result = workflow._should_search_papers(state)
        assert result is True

    def test_should_search_papers_false(self, workflow):
        """Test routing when paper search is disabled."""
        state = {"search_papers": False}
        result = workflow._should_search_papers(state)
        assert result is False

    def test_should_search_papers_default(self, workflow):
        """Test routing with default (missing) search_papers."""
        state = {}
        result = workflow._should_search_papers(state)
        assert result is True  # Default is True


# =============================================================================
# Full Workflow Execution Tests
# =============================================================================


class TestFullWorkflowExecution:
    """Tests for complete workflow execution."""

    @pytest.mark.asyncio
    async def test_full_workflow_with_search(self, workflow, sample_initial_state):
        """Test complete workflow execution with paper search."""
        result = await workflow.run(
            topic=sample_initial_state["topic"],
            research_question=sample_initial_state["research_question"],
            request_id="test-full-1",
            research_type="systematic_review",
            discipline="education",
            academic_level="graduate",
            paper_limit=10,
            use_llm=False,
            search_papers=True,
        )

        # Verify final output structure
        assert "topic" in result
        assert "research_question" in result
        assert "topic_analysis" in result
        assert "papers" in result
        assert "methodology_recommendations" in result
        assert "quality_score" in result

    @pytest.mark.asyncio
    async def test_full_workflow_without_search(self, workflow, sample_initial_state):
        """Test complete workflow execution without paper search."""
        result = await workflow.run(
            topic=sample_initial_state["topic"],
            research_question=sample_initial_state["research_question"],
            request_id="test-full-2",
            use_llm=False,
            search_papers=False,  # Skip paper search
        )

        # Should still complete but with no papers
        assert "topic" in result
        assert "methodology_recommendations" in result

    @pytest.mark.asyncio
    async def test_workflow_handles_search_errors(self, workflow, sample_initial_state):
        """Test workflow handles search errors gracefully."""
        # Configure search to fail
        workflow.search_client.search = AsyncMock(side_effect=Exception("Search API unavailable"))

        result = await workflow.run(
            topic=sample_initial_state["topic"],
            research_question=sample_initial_state["research_question"],
            request_id="test-error-1",
            use_llm=False,
            search_papers=True,
        )

        # Workflow should complete despite search errors
        assert "topic" in result
        # Errors should be recorded
        # Note: errors are accumulated in state


# =============================================================================
# State Management Tests
# =============================================================================


class TestStateManagement:
    """Tests for workflow state management."""

    def test_initial_state_structure(self, sample_initial_state):
        """Test that initial state has correct structure."""
        required_keys = ["topic", "research_question", "research_type"]
        for key in required_keys:
            assert key in sample_initial_state

    @pytest.mark.asyncio
    async def test_state_accumulates_errors(self, workflow):
        """Test that errors accumulate in state."""
        # Configure search to fail multiple times
        workflow.search_client.search = AsyncMock(side_effect=Exception("Search failed"))

        state = {
            "topic": "Test",
            "research_question": "Test question",
            "search_queries": ["query1", "query2", "query3"],
            "paper_limit": 10,
            "errors": [],
        }

        result = await workflow._search_papers(state)

        # Errors should be accumulated
        assert len(result.get("errors", [])) > 0


# =============================================================================
# Rule-Based Fallback Tests
# =============================================================================


class TestRuleBasedFallbacks:
    """Tests for rule-based analysis fallbacks."""

    def test_rule_based_topic_analysis(self, workflow):
        """Test rule-based topic analysis."""
        result = workflow._rule_based_topic_analysis(
            topic="artificial intelligence machine learning",
            research_question="How does AI work?",
        )

        assert "topic" in result
        assert "key_concepts" in result
        assert len(result["key_concepts"]) > 0
        # Should filter stop words
        assert "the" not in result["key_concepts"]

    def test_rule_based_paper_analysis(self, workflow):
        """Test rule-based paper analysis."""
        papers = [
            {"title": "Paper 1", "year": 2023},
            {"title": "Paper 2", "year": 2024},
            {"title": "Paper 3", "year": 2022},
        ]

        result = workflow._rule_based_paper_analysis(papers)

        assert result["papers_analyzed"] == 3
        assert "2022-2024" in result["year_range"]

    def test_rule_based_paper_analysis_empty(self, workflow):
        """Test rule-based paper analysis with no papers."""
        result = workflow._rule_based_paper_analysis([])

        assert result["papers_analyzed"] == 0
        assert "N/A" in result["year_range"]
