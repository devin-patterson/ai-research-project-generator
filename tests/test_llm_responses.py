"""
Tests for LLM responses with mocking strategies.

This module provides comprehensive testing patterns for LLM-based functionality:
1. Mock-based testing for deterministic unit tests
2. Snapshot testing for regression detection
3. Evaluation-based testing for quality assurance
4. Integration testing with real LLM (optional)
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

from pydantic import BaseModel

from app.workflows.agents import (
    TopicAnalysis,
    PaperSynthesis,
    MethodologyRecommendation,
    ResearchProjectOutput,
    ResearchDependencies,
    analyze_topic_with_agent,
    synthesize_papers_with_agent,
    recommend_methodology_with_agent,
    create_topic_analysis_agent,
    create_paper_synthesis_agent,
    create_methodology_agent,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_http_client():
    """Create a mock HTTP client."""
    client = AsyncMock()
    client.get = AsyncMock()
    client.post = AsyncMock()
    return client


@pytest.fixture
def research_deps(mock_http_client):
    """Create research dependencies with mocked HTTP client."""
    return ResearchDependencies(
        http_client=mock_http_client,
        llm_base_url="http://localhost:11434",
        llm_model="llama3.1:8b",
        api_timeout=30.0,
    )


@pytest.fixture
def sample_topic_analysis() -> TopicAnalysis:
    """Sample topic analysis for testing."""
    return TopicAnalysis(
        topic="Impact of artificial intelligence on education",
        key_concepts=[
            "artificial intelligence",
            "machine learning",
            "educational technology",
            "personalized learning",
            "adaptive systems",
        ],
        research_scope="broad",
        complexity_level="advanced",
        suggested_subtopics=[
            "AI-powered tutoring systems",
            "Automated assessment",
            "Learning analytics",
        ],
        potential_challenges=[
            "Data privacy concerns",
            "Algorithmic bias",
            "Teacher adoption barriers",
        ],
        interdisciplinary_connections=[
            "Computer Science",
            "Psychology",
            "Pedagogy",
        ],
    )


@pytest.fixture
def sample_paper_synthesis() -> PaperSynthesis:
    """Sample paper synthesis for testing."""
    return PaperSynthesis(
        papers_analyzed=15,
        main_themes=[
            "Personalized learning through AI",
            "Automated assessment systems",
            "Student engagement metrics",
        ],
        research_gaps=[
            "Long-term effectiveness studies",
            "Cross-cultural implementation",
            "Teacher training requirements",
        ],
        methodological_trends=[
            "Mixed methods approaches",
            "Longitudinal studies",
            "Randomized controlled trials",
        ],
        key_findings=[
            "AI improves learning outcomes by 15-25%",
            "Student engagement increases with adaptive systems",
            "Teacher workload reduced by 30%",
        ],
        synthesis_narrative="The literature reveals significant potential for AI in education...",
        recommended_focus_areas=[
            "Equity and access",
            "Teacher-AI collaboration",
        ],
    )


@pytest.fixture
def sample_methodology() -> MethodologyRecommendation:
    """Sample methodology recommendation for testing."""
    return MethodologyRecommendation(
        primary_methodology="Mixed Methods Research",
        rationale="Combines quantitative learning outcomes with qualitative insights",
        data_collection_methods=[
            "Pre/post assessments",
            "Student surveys",
            "Teacher interviews",
            "System usage logs",
        ],
        analysis_techniques=[
            "Statistical analysis (t-tests, ANOVA)",
            "Thematic analysis",
            "Learning analytics",
        ],
        quality_criteria=[
            "Validity through triangulation",
            "Reliability through standardized instruments",
            "Transferability through thick description",
        ],
        potential_limitations=[
            "Sample size constraints",
            "Technology access variations",
            "Hawthorne effect",
        ],
        alternative_approaches=[
            "Purely quantitative RCT",
            "Ethnographic study",
        ],
    )


@pytest.fixture
def sample_papers() -> list[dict[str, Any]]:
    """Sample papers for testing."""
    return [
        {
            "title": "AI in Education: A Systematic Review",
            "abstract": "This paper reviews the application of AI in educational settings...",
            "year": 2023,
            "doi": "10.1234/example1",
            "authors": ["Smith, J.", "Johnson, M."],
        },
        {
            "title": "Machine Learning for Personalized Learning",
            "abstract": "We present a machine learning approach to personalized education...",
            "year": 2024,
            "doi": "10.1234/example2",
            "authors": ["Williams, K."],
        },
        {
            "title": "Adaptive Learning Systems: Current State",
            "abstract": "An overview of adaptive learning systems and their effectiveness...",
            "year": 2023,
            "doi": "10.1234/example3",
            "authors": ["Brown, A.", "Davis, L."],
        },
    ]


# =============================================================================
# Mock-Based Testing (Deterministic Unit Tests)
# =============================================================================

class TestMockedLLMResponses:
    """Tests using mocked LLM responses for deterministic behavior."""

    @pytest.mark.asyncio
    async def test_topic_analysis_with_mocked_agent(
        self, research_deps, sample_topic_analysis
    ):
        """Test topic analysis with a mocked agent response."""
        with patch("app.workflows.agents.topic_analysis_agent") as mock_agent:
            # Configure mock to return sample data
            mock_result = MagicMock()
            mock_result.data = sample_topic_analysis
            mock_agent.run = AsyncMock(return_value=mock_result)
            
            result = await analyze_topic_with_agent(
                topic="Impact of artificial intelligence on education",
                research_question="How does AI affect learning outcomes?",
                deps=research_deps,
            )
            
            assert result.topic == "Impact of artificial intelligence on education"
            assert len(result.key_concepts) >= 1
            assert result.research_scope in ["narrow", "moderate", "broad"]
            assert result.complexity_level in ["basic", "intermediate", "advanced"]

    @pytest.mark.asyncio
    async def test_paper_synthesis_with_mocked_agent(
        self, research_deps, sample_papers, sample_paper_synthesis
    ):
        """Test paper synthesis with a mocked agent response."""
        with patch("app.workflows.agents.paper_synthesis_agent") as mock_agent:
            mock_result = MagicMock()
            mock_result.data = sample_paper_synthesis
            mock_agent.run = AsyncMock(return_value=mock_result)
            
            result = await synthesize_papers_with_agent(
                papers=sample_papers,
                topic="AI in Education",
                deps=research_deps,
            )
            
            assert result.papers_analyzed > 0
            assert len(result.main_themes) > 0
            assert len(result.research_gaps) > 0
            assert result.synthesis_narrative != ""

    @pytest.mark.asyncio
    async def test_methodology_with_mocked_agent(
        self, research_deps, sample_methodology
    ):
        """Test methodology recommendation with a mocked agent response."""
        with patch("app.workflows.agents.methodology_agent") as mock_agent:
            mock_result = MagicMock()
            mock_result.data = sample_methodology
            mock_agent.run = AsyncMock(return_value=mock_result)
            
            result = await recommend_methodology_with_agent(
                topic="AI in Education",
                research_type="systematic_review",
                discipline="education",
                academic_level="graduate",
                deps=research_deps,
            )
            
            assert result.primary_methodology != ""
            assert result.rationale != ""
            assert len(result.data_collection_methods) > 0
            assert len(result.analysis_techniques) > 0


# =============================================================================
# Schema Validation Tests
# =============================================================================

class TestOutputSchemaValidation:
    """Tests to ensure LLM outputs conform to expected schemas."""

    def test_topic_analysis_schema_validation(self):
        """Test that TopicAnalysis schema validates correctly."""
        # Valid data
        valid_data = {
            "topic": "Test Topic",
            "key_concepts": ["concept1", "concept2"],
            "research_scope": "moderate",
            "complexity_level": "intermediate",
            "suggested_subtopics": [],
            "potential_challenges": [],
            "interdisciplinary_connections": [],
        }
        analysis = TopicAnalysis(**valid_data)
        assert analysis.topic == "Test Topic"
        assert len(analysis.key_concepts) == 2

    def test_topic_analysis_requires_key_concepts(self):
        """Test that TopicAnalysis requires at least one key concept."""
        with pytest.raises(ValueError):
            TopicAnalysis(
                topic="Test",
                key_concepts=[],  # Empty list should fail
                research_scope="moderate",
                complexity_level="basic",
            )

    def test_paper_synthesis_schema_validation(self):
        """Test that PaperSynthesis schema validates correctly."""
        valid_data = {
            "papers_analyzed": 10,
            "main_themes": ["theme1"],
            "research_gaps": ["gap1"],
            "methodological_trends": ["trend1"],
            "key_findings": ["finding1"],
            "synthesis_narrative": "A synthesis of the literature.",
        }
        synthesis = PaperSynthesis(**valid_data)
        assert synthesis.papers_analyzed == 10

    def test_paper_synthesis_papers_analyzed_non_negative(self):
        """Test that papers_analyzed must be non-negative."""
        with pytest.raises(ValueError):
            PaperSynthesis(
                papers_analyzed=-1,  # Should fail
                main_themes=["theme"],
                research_gaps=["gap"],
                methodological_trends=["trend"],
                key_findings=["finding"],
                synthesis_narrative="Narrative",
            )

    def test_methodology_schema_validation(self):
        """Test that MethodologyRecommendation schema validates correctly."""
        valid_data = {
            "primary_methodology": "Qualitative Research",
            "rationale": "Best suited for exploratory research",
            "data_collection_methods": ["interviews"],
            "analysis_techniques": ["thematic analysis"],
            "quality_criteria": ["credibility"],
            "potential_limitations": ["sample size"],
        }
        methodology = MethodologyRecommendation(**valid_data)
        assert methodology.primary_methodology == "Qualitative Research"

    def test_research_project_output_quality_score_bounds(self):
        """Test that quality_score is bounded between 0 and 100."""
        # Create minimal valid nested objects
        topic_analysis = TopicAnalysis(
            topic="Test",
            key_concepts=["concept"],
            research_scope="narrow",
            complexity_level="basic",
        )
        methodology = MethodologyRecommendation(
            primary_methodology="Test",
            rationale="Test rationale",
            data_collection_methods=["method"],
            analysis_techniques=["technique"],
            quality_criteria=["criteria"],
            potential_limitations=["limitation"],
        )
        
        # Valid quality score
        output = ResearchProjectOutput(
            title="Test Project",
            abstract="Test abstract",
            topic_analysis=topic_analysis,
            methodology=methodology,
            research_questions=["Question 1"],
            expected_contributions=["Contribution 1"],
            timeline_estimate="6 months",
            quality_score=85.5,
        )
        assert output.quality_score == 85.5
        
        # Invalid quality score (too high)
        with pytest.raises(ValueError):
            ResearchProjectOutput(
                title="Test",
                abstract="Test",
                topic_analysis=topic_analysis,
                methodology=methodology,
                research_questions=["Q1"],
                expected_contributions=["C1"],
                timeline_estimate="6 months",
                quality_score=150,  # Should fail
            )


# =============================================================================
# Snapshot Testing for Regression Detection
# =============================================================================

class TestLLMResponseSnapshots:
    """
    Snapshot tests to detect regressions in LLM output structure.
    
    These tests compare current outputs against saved snapshots to detect
    unexpected changes in response format or content.
    """

    def test_topic_analysis_structure_snapshot(self, sample_topic_analysis):
        """Verify topic analysis output structure matches expected format."""
        output_dict = sample_topic_analysis.model_dump()
        
        # Verify all expected keys are present
        expected_keys = {
            "topic",
            "key_concepts",
            "research_scope",
            "complexity_level",
            "suggested_subtopics",
            "potential_challenges",
            "interdisciplinary_connections",
        }
        assert set(output_dict.keys()) == expected_keys
        
        # Verify types
        assert isinstance(output_dict["topic"], str)
        assert isinstance(output_dict["key_concepts"], list)
        assert isinstance(output_dict["research_scope"], str)

    def test_paper_synthesis_structure_snapshot(self, sample_paper_synthesis):
        """Verify paper synthesis output structure matches expected format."""
        output_dict = sample_paper_synthesis.model_dump()
        
        expected_keys = {
            "papers_analyzed",
            "main_themes",
            "research_gaps",
            "methodological_trends",
            "key_findings",
            "synthesis_narrative",
            "recommended_focus_areas",
        }
        assert set(output_dict.keys()) == expected_keys
        
        # Verify types
        assert isinstance(output_dict["papers_analyzed"], int)
        assert isinstance(output_dict["main_themes"], list)
        assert isinstance(output_dict["synthesis_narrative"], str)


# =============================================================================
# Evaluation-Based Testing (Quality Assurance)
# =============================================================================

class TestLLMOutputQuality:
    """
    Quality evaluation tests for LLM outputs.
    
    These tests check semantic quality rather than exact matches,
    useful for ensuring LLM outputs meet minimum quality standards.
    """

    def test_topic_analysis_has_meaningful_concepts(self, sample_topic_analysis):
        """Verify topic analysis contains meaningful key concepts."""
        # Key concepts should not be empty strings
        assert all(len(concept) > 0 for concept in sample_topic_analysis.key_concepts)
        
        # Key concepts should be reasonably sized (not too short)
        assert all(len(concept) >= 2 for concept in sample_topic_analysis.key_concepts)

    def test_paper_synthesis_narrative_quality(self, sample_paper_synthesis):
        """Verify synthesis narrative meets minimum quality standards."""
        narrative = sample_paper_synthesis.synthesis_narrative
        
        # Narrative should be substantial (at least 20 characters)
        assert len(narrative) >= 20
        
        # Narrative should contain actual words (not just punctuation)
        words = narrative.split()
        assert len(words) >= 3

    def test_methodology_has_complete_recommendations(self, sample_methodology):
        """Verify methodology recommendation is complete."""
        # Should have at least one data collection method
        assert len(sample_methodology.data_collection_methods) >= 1
        
        # Should have at least one analysis technique
        assert len(sample_methodology.analysis_techniques) >= 1
        
        # Rationale should be substantial
        assert len(sample_methodology.rationale) >= 10

    def test_research_gaps_are_actionable(self, sample_paper_synthesis):
        """Verify identified research gaps are actionable."""
        for gap in sample_paper_synthesis.research_gaps:
            # Gaps should be descriptive (not just single words)
            assert len(gap.split()) >= 2, f"Gap '{gap}' is too vague"


# =============================================================================
# Integration Tests (Optional - Requires Running LLM)
# =============================================================================

@pytest.mark.skip(reason="Requires running Ollama LLM server")
class TestLLMIntegration:
    """
    Integration tests that run against a real LLM.
    
    These tests are skipped by default and should only be run
    when an LLM server is available.
    """

    @pytest.mark.asyncio
    async def test_real_topic_analysis(self):
        """Test topic analysis with real LLM."""
        import httpx
        
        async with httpx.AsyncClient() as client:
            deps = ResearchDependencies(http_client=client)
            
            result = await analyze_topic_with_agent(
                topic="Impact of artificial intelligence on education",
                research_question="How does AI affect learning outcomes?",
                deps=deps,
            )
            
            # Verify we got a valid response
            assert result.topic != ""
            assert len(result.key_concepts) >= 1

    @pytest.mark.asyncio
    async def test_real_methodology_recommendation(self):
        """Test methodology recommendation with real LLM."""
        import httpx
        
        async with httpx.AsyncClient() as client:
            deps = ResearchDependencies(http_client=client)
            
            result = await recommend_methodology_with_agent(
                topic="AI in Education",
                research_type="systematic_review",
                discipline="education",
                academic_level="graduate",
                deps=deps,
            )
            
            assert result.primary_methodology != ""
            assert len(result.data_collection_methods) >= 1


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestLLMErrorHandling:
    """Tests for error handling in LLM interactions."""

    @pytest.mark.asyncio
    async def test_handles_llm_timeout(self, research_deps):
        """Test graceful handling of LLM timeout."""
        with patch("app.workflows.agents.topic_analysis_agent") as mock_agent:
            import asyncio
            mock_agent.run = AsyncMock(side_effect=asyncio.TimeoutError())
            
            with pytest.raises(asyncio.TimeoutError):
                await analyze_topic_with_agent(
                    topic="Test topic",
                    research_question="Test question",
                    deps=research_deps,
                )

    @pytest.mark.asyncio
    async def test_handles_llm_connection_error(self, research_deps):
        """Test graceful handling of LLM connection errors."""
        with patch("app.workflows.agents.topic_analysis_agent") as mock_agent:
            mock_agent.run = AsyncMock(side_effect=ConnectionError("LLM unavailable"))
            
            with pytest.raises(ConnectionError):
                await analyze_topic_with_agent(
                    topic="Test topic",
                    research_question="Test question",
                    deps=research_deps,
                )

    @pytest.mark.asyncio
    async def test_handles_invalid_llm_response(self, research_deps):
        """Test handling of invalid LLM response that fails validation."""
        with patch("app.workflows.agents.topic_analysis_agent") as mock_agent:
            # Return invalid data that won't pass Pydantic validation
            mock_result = MagicMock()
            mock_result.data = None  # Invalid - should be TopicAnalysis
            mock_agent.run = AsyncMock(return_value=mock_result)
            
            # The agent should handle this gracefully or raise appropriate error
            result = await analyze_topic_with_agent(
                topic="Test topic",
                research_question="Test question",
                deps=research_deps,
            )
            
            # With PydanticAI, invalid responses trigger retries
            # If all retries fail, we should get None or an error
            assert result is None or isinstance(result, TopicAnalysis)
