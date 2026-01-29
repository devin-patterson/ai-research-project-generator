"""
Tests for Pydantic schemas.
"""

import pytest
from pydantic import ValidationError

from app.schemas.research import (
    ResearchRequest,
    ResearchType,
    AcademicLevel,
)


class TestResearchRequest:
    """Tests for ResearchRequest schema."""

    def test_valid_request(self):
        """Valid request should be accepted."""
        request = ResearchRequest(
            topic="Impact of artificial intelligence on education",
            research_question="How does AI affect learning outcomes?",
        )
        assert request.topic == "Impact of artificial intelligence on education"
        assert request.research_question == "How does AI affect learning outcomes?"

    def test_default_values(self):
        """Default values should be set correctly."""
        request = ResearchRequest(
            topic="Impact of artificial intelligence on education",
            research_question="How does AI affect learning outcomes?",
        )
        assert request.research_type == ResearchType.SYSTEMATIC_REVIEW
        assert request.academic_level == AcademicLevel.GRADUATE
        assert request.discipline == "general"
        assert request.search_papers is True
        assert request.paper_limit == 20
        assert request.use_llm is True

    def test_topic_min_length(self):
        """Topic should have minimum length of 10."""
        with pytest.raises(ValidationError) as exc_info:
            ResearchRequest(
                topic="AI",
                research_question="How does AI affect learning outcomes?",
            )
        assert "String should have at least 10 characters" in str(exc_info.value)

    def test_topic_max_length(self):
        """Topic should have maximum length of 2000."""
        with pytest.raises(ValidationError) as exc_info:
            ResearchRequest(
                topic="A" * 2001,
                research_question="How does AI affect learning outcomes?",
            )
        assert "String should have at most 2000 characters" in str(exc_info.value)

    def test_research_question_min_length(self):
        """Research question should have minimum length of 10."""
        with pytest.raises(ValidationError) as exc_info:
            ResearchRequest(
                topic="Impact of artificial intelligence on education",
                research_question="AI?",
            )
        assert "String should have at least 10 characters" in str(exc_info.value)

    def test_paper_limit_min(self):
        """Paper limit should be at least 1."""
        with pytest.raises(ValidationError) as exc_info:
            ResearchRequest(
                topic="Impact of artificial intelligence on education",
                research_question="How does AI affect learning outcomes?",
                paper_limit=0,
            )
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_paper_limit_max(self):
        """Paper limit should be at most 100."""
        with pytest.raises(ValidationError) as exc_info:
            ResearchRequest(
                topic="Impact of artificial intelligence on education",
                research_question="How does AI affect learning outcomes?",
                paper_limit=101,
            )
        assert "less than or equal to 100" in str(exc_info.value)

    def test_year_range_validation(self):
        """Year end should be >= year start."""
        with pytest.raises(ValidationError) as exc_info:
            ResearchRequest(
                topic="Impact of artificial intelligence on education",
                research_question="How does AI affect learning outcomes?",
                year_start=2025,
                year_end=2020,
            )
        assert "year_end must be greater than or equal to year_start" in str(exc_info.value)

    def test_valid_year_range(self):
        """Valid year range should be accepted."""
        request = ResearchRequest(
            topic="Impact of artificial intelligence on education",
            research_question="How does AI affect learning outcomes?",
            year_start=2020,
            year_end=2025,
        )
        assert request.year_start == 2020
        assert request.year_end == 2025

    def test_research_type_enum(self):
        """Research type should accept valid enum values."""
        request = ResearchRequest(
            topic="Impact of artificial intelligence on education",
            research_question="How does AI affect learning outcomes?",
            research_type="meta_analysis",
        )
        assert request.research_type == ResearchType.META_ANALYSIS

    def test_invalid_research_type(self):
        """Invalid research type should be rejected."""
        with pytest.raises(ValidationError):
            ResearchRequest(
                topic="Impact of artificial intelligence on education",
                research_question="How does AI affect learning outcomes?",
                research_type="invalid_type",
            )

    def test_academic_level_enum(self):
        """Academic level should accept valid enum values."""
        request = ResearchRequest(
            topic="Impact of artificial intelligence on education",
            research_question="How does AI affect learning outcomes?",
            academic_level="doctoral",
        )
        assert request.academic_level == AcademicLevel.DOCTORAL
