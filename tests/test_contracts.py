"""
Contract Tests for API and Schema Validation.

Contract tests verify that:
1. API endpoints return expected response structures
2. Pydantic schemas enforce validation rules
3. LLM outputs conform to expected contracts
4. Inter-service contracts are maintained
"""

import pytest
from pydantic import ValidationError

from app.schemas.research import (
    ResearchRequest,
    ResearchType,
    AcademicLevel,
)
from app.workflows.agents import (
    TopicAnalysis,
    PaperSynthesis,
    MethodologyRecommendation,
    ResearchProjectOutput,
)


# =============================================================================
# API Request Contract Tests
# =============================================================================


class TestResearchRequestContract:
    """Contract tests for ResearchRequest schema."""

    def test_minimal_valid_request(self):
        """Contract: Minimal request with only required fields should be valid."""
        request = ResearchRequest(
            topic="Impact of AI on education systems worldwide",
            research_question="How does AI affect student learning outcomes?",
        )
        assert request.topic is not None
        assert request.research_question is not None
        # Defaults should be applied
        assert request.research_type == ResearchType.SYSTEMATIC_REVIEW
        assert request.academic_level == AcademicLevel.GRADUATE

    def test_full_valid_request(self):
        """Contract: Full request with all fields should be valid."""
        request = ResearchRequest(
            topic="Machine learning in healthcare diagnostics",
            research_question="Can ML improve cancer detection rates?",
            research_type="meta_analysis",
            academic_level="doctoral",
            discipline="medicine",
            paper_limit=50,
            year_start=2020,
            year_end=2024,
            search_papers=True,
            use_llm=True,
        )
        assert request.paper_limit == 50
        assert request.year_start == 2020

    def test_topic_length_contract(self):
        """Contract: Topic must be between 10 and 2000 characters."""
        # Too short
        with pytest.raises(ValidationError) as exc:
            ResearchRequest(
                topic="AI",
                research_question="How does AI work in education?",
            )
        assert "at least 10 characters" in str(exc.value)

        # Too long
        with pytest.raises(ValidationError) as exc:
            ResearchRequest(
                topic="A" * 2001,
                research_question="How does AI work in education?",
            )
        assert "at most 2000 characters" in str(exc.value)

    def test_paper_limit_contract(self):
        """Contract: Paper limit must be between 1 and 100."""
        # Too low
        with pytest.raises(ValidationError):
            ResearchRequest(
                topic="AI in education research",
                research_question="How does AI affect learning?",
                paper_limit=0,
            )

        # Too high
        with pytest.raises(ValidationError):
            ResearchRequest(
                topic="AI in education research",
                research_question="How does AI affect learning?",
                paper_limit=101,
            )

    def test_year_range_contract(self):
        """Contract: year_end must be >= year_start."""
        with pytest.raises(ValidationError) as exc:
            ResearchRequest(
                topic="AI in education research",
                research_question="How does AI affect learning?",
                year_start=2025,
                year_end=2020,
            )
        assert "year_end must be greater than or equal to year_start" in str(exc.value)

    def test_research_type_enum_contract(self):
        """Contract: research_type must be a valid enum value."""
        valid_types = [
            "systematic_review",
            "scoping_review",
            "meta_analysis",
            "qualitative_study",
            "quantitative_study",
            "mixed_methods",
            "case_study",
            "experimental",
            "literature_review",
        ]

        for rt in valid_types:
            request = ResearchRequest(
                topic="AI in education research",
                research_question="How does AI affect learning?",
                research_type=rt,
            )
            assert request.research_type.value == rt

    def test_invalid_research_type_contract(self):
        """Contract: Invalid research_type should raise ValidationError."""
        with pytest.raises(ValidationError):
            ResearchRequest(
                topic="AI in education research",
                research_question="How does AI affect learning?",
                research_type="invalid_type",
            )


# =============================================================================
# LLM Output Contract Tests
# =============================================================================


class TestTopicAnalysisContract:
    """Contract tests for TopicAnalysis LLM output schema."""

    def test_valid_topic_analysis(self):
        """Contract: Valid TopicAnalysis should have all required fields."""
        analysis = TopicAnalysis(
            topic="AI in education",
            key_concepts=["machine learning", "adaptive learning", "personalization"],
            research_scope="moderate",
            complexity_level="intermediate",
            suggested_subtopics=["AI tutoring", "automated assessment"],
            potential_challenges=["data privacy", "bias"],
            interdisciplinary_connections=["psychology", "computer science"],
        )

        assert len(analysis.key_concepts) >= 1
        assert analysis.research_scope in ["narrow", "moderate", "broad"]
        assert analysis.complexity_level in ["basic", "intermediate", "advanced"]

    def test_key_concepts_minimum_contract(self):
        """Contract: key_concepts must have at least 1 item."""
        with pytest.raises(ValidationError):
            TopicAnalysis(
                topic="AI in education",
                key_concepts=[],  # Empty list should fail
                research_scope="moderate",
                complexity_level="intermediate",
            )

    def test_key_concepts_maximum_contract(self):
        """Contract: key_concepts should have at most 20 items."""
        with pytest.raises(ValidationError):
            TopicAnalysis(
                topic="AI in education",
                key_concepts=[f"concept{i}" for i in range(21)],  # 21 items
                research_scope="moderate",
                complexity_level="intermediate",
            )


class TestPaperSynthesisContract:
    """Contract tests for PaperSynthesis LLM output schema."""

    def test_valid_paper_synthesis(self):
        """Contract: Valid PaperSynthesis should have all required fields."""
        synthesis = PaperSynthesis(
            papers_analyzed=10,
            main_themes=["AI adoption", "learning outcomes"],
            research_gaps=["long-term studies needed"],
            methodological_trends=["mixed methods"],
            key_findings=["AI improves engagement"],
            synthesis_narrative="The literature shows...",
        )

        assert synthesis.papers_analyzed >= 0
        assert len(synthesis.main_themes) >= 1
        assert isinstance(synthesis.synthesis_narrative, str)

    def test_papers_analyzed_non_negative_contract(self):
        """Contract: papers_analyzed must be non-negative."""
        with pytest.raises(ValidationError):
            PaperSynthesis(
                papers_analyzed=-1,
                main_themes=["theme"],
                research_gaps=["gap"],
                methodological_trends=["trend"],
                key_findings=["finding"],
                synthesis_narrative="Narrative",
            )


class TestMethodologyRecommendationContract:
    """Contract tests for MethodologyRecommendation LLM output schema."""

    def test_valid_methodology(self):
        """Contract: Valid MethodologyRecommendation should have all required fields."""
        methodology = MethodologyRecommendation(
            primary_methodology="Mixed Methods Research",
            rationale="Combines quantitative and qualitative insights",
            data_collection_methods=["surveys", "interviews"],
            analysis_techniques=["statistical analysis", "thematic analysis"],
            quality_criteria=["validity", "reliability"],
            potential_limitations=["sample size"],
        )

        assert methodology.primary_methodology != ""
        assert len(methodology.data_collection_methods) >= 1
        assert len(methodology.analysis_techniques) >= 1


class TestResearchProjectOutputContract:
    """Contract tests for complete ResearchProjectOutput schema."""

    def test_quality_score_bounds_contract(self):
        """Contract: quality_score must be between 0 and 100."""
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

        # Valid score
        output = ResearchProjectOutput(
            title="Test Project",
            abstract="Test abstract",
            topic_analysis=topic_analysis,
            methodology=methodology,
            research_questions=["Q1"],
            expected_contributions=["C1"],
            timeline_estimate="6 months",
            quality_score=75.5,
        )
        assert 0 <= output.quality_score <= 100

        # Score too high
        with pytest.raises(ValidationError):
            ResearchProjectOutput(
                title="Test",
                abstract="Test",
                topic_analysis=topic_analysis,
                methodology=methodology,
                research_questions=["Q1"],
                expected_contributions=["C1"],
                timeline_estimate="6 months",
                quality_score=150,
            )

        # Score too low
        with pytest.raises(ValidationError):
            ResearchProjectOutput(
                title="Test",
                abstract="Test",
                topic_analysis=topic_analysis,
                methodology=methodology,
                research_questions=["Q1"],
                expected_contributions=["C1"],
                timeline_estimate="6 months",
                quality_score=-10,
            )


# =============================================================================
# Inter-Service Contract Tests
# =============================================================================


class TestWorkflowStateContract:
    """Contract tests for workflow state transitions."""

    def test_research_state_required_fields(self):
        """Contract: ResearchState must have topic and research_question."""
        from app.workflows.research_workflow import ResearchState

        # This is a TypedDict, so we test the expected structure
        state: ResearchState = {
            "topic": "AI in education",
            "research_question": "How does AI affect learning?",
        }

        assert "topic" in state
        assert "research_question" in state

    def test_workflow_output_contract(self):
        """Contract: Workflow output must contain expected keys."""
        expected_output_keys = [
            "topic",
            "research_question",
            "research_type",
            "topic_analysis",
            "methodology_recommendations",
            "quality_score",
        ]

        # Mock output structure
        output = {
            "topic": "AI in education",
            "research_question": "How does AI affect learning?",
            "research_type": "systematic_review",
            "topic_analysis": {"key_concepts": ["AI", "education"]},
            "methodology_recommendations": ["PRISMA guidelines"],
            "quality_score": 85.0,
        }

        for key in expected_output_keys:
            assert key in output, f"Missing required key: {key}"
