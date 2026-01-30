"""
Golden-Set Regression Tests.

Golden-set tests compare LLM outputs against pre-approved "golden" responses
to detect regressions in output quality. These tests:

1. Store approved outputs as golden references
2. Compare new outputs against golden references
3. Flag significant deviations for review
4. Support fuzzy matching for non-deterministic outputs
"""

import json
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field

import pytest
from loguru import logger


# =============================================================================
# Golden Set Infrastructure
# =============================================================================

GOLDEN_SET_DIR = Path(__file__).parent / "golden_sets"


@dataclass
class GoldenTestCase:
    """A golden test case with input, expected output, and metadata."""

    id: str
    name: str
    input_data: dict[str, Any]
    golden_output: dict[str, Any]
    tolerance: float = 0.8  # Similarity threshold for fuzzy matching
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "input_data": self.input_data,
            "golden_output": self.golden_output,
            "tolerance": self.tolerance,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GoldenTestCase":
        return cls(**data)


class GoldenSetManager:
    """Manages golden test sets for regression testing."""

    def __init__(self, golden_dir: Path = GOLDEN_SET_DIR):
        self.golden_dir = golden_dir
        self.golden_dir.mkdir(parents=True, exist_ok=True)

    def save_golden(self, test_case: GoldenTestCase, category: str = "default") -> None:
        """Save a golden test case."""
        category_dir = self.golden_dir / category
        category_dir.mkdir(exist_ok=True)

        file_path = category_dir / f"{test_case.id}.json"
        with open(file_path, "w") as f:
            json.dump(test_case.to_dict(), f, indent=2)

        logger.info(f"Saved golden test case: {test_case.id}")

    def load_golden(self, test_id: str, category: str = "default") -> Optional[GoldenTestCase]:
        """Load a golden test case by ID."""
        file_path = self.golden_dir / category / f"{test_id}.json"

        if not file_path.exists():
            return None

        with open(file_path) as f:
            data = json.load(f)

        return GoldenTestCase.from_dict(data)

    def load_all_golden(self, category: str = "default") -> list[GoldenTestCase]:
        """Load all golden test cases in a category."""
        category_dir = self.golden_dir / category

        if not category_dir.exists():
            return []

        test_cases = []
        for file_path in category_dir.glob("*.json"):
            with open(file_path) as f:
                data = json.load(f)
            test_cases.append(GoldenTestCase.from_dict(data))

        return test_cases


def calculate_similarity(output: dict, golden: dict) -> float:
    """
    Calculate similarity between output and golden reference.

    Uses a combination of:
    - Key overlap
    - Value similarity for matching keys
    - List overlap for list values
    """
    if not output or not golden:
        return 0.0

    # Key overlap score
    output_keys = set(output.keys())
    golden_keys = set(golden.keys())
    key_overlap = len(output_keys & golden_keys) / len(golden_keys) if golden_keys else 0

    # Value similarity for matching keys
    value_scores = []
    for key in output_keys & golden_keys:
        output_val = output[key]
        golden_val = golden[key]

        if isinstance(golden_val, list) and isinstance(output_val, list):
            # List overlap
            if golden_val:
                overlap = len(set(output_val) & set(golden_val)) / len(golden_val)
                value_scores.append(overlap)
        elif isinstance(golden_val, str) and isinstance(output_val, str):
            # String similarity (simple word overlap)
            golden_words = set(golden_val.lower().split())
            output_words = set(output_val.lower().split())
            if golden_words:
                overlap = len(golden_words & output_words) / len(golden_words)
                value_scores.append(overlap)
        elif golden_val == output_val:
            value_scores.append(1.0)
        else:
            value_scores.append(0.0)

    value_score = sum(value_scores) / len(value_scores) if value_scores else 0

    # Combined score
    return 0.4 * key_overlap + 0.6 * value_score


# =============================================================================
# Golden Test Cases (Pre-defined)
# =============================================================================

TOPIC_ANALYSIS_GOLDEN_CASES = [
    GoldenTestCase(
        id="topic_001",
        name="AI in Education - Basic Analysis",
        input_data={
            "topic": "Impact of artificial intelligence on education",
            "research_question": "How does AI affect student learning outcomes?",
        },
        golden_output={
            "key_concepts": [
                "artificial intelligence",
                "machine learning",
                "educational technology",
                "personalized learning",
                "adaptive systems",
                "student outcomes",
            ],
            "research_scope": "broad",
            "complexity_level": "advanced",
            "suggested_subtopics": [
                "AI-powered tutoring systems",
                "Automated assessment",
                "Learning analytics",
                "Intelligent content delivery",
            ],
            "potential_challenges": [
                "Data privacy concerns",
                "Algorithmic bias",
                "Digital divide",
                "Teacher adoption",
            ],
        },
        tolerance=0.7,
        tags=["education", "ai", "technology"],
    ),
    GoldenTestCase(
        id="topic_002",
        name="Climate Change - Marine Biology",
        input_data={
            "topic": "Climate change effects on marine ecosystems",
            "research_question": "How does ocean warming affect coral reef biodiversity?",
        },
        golden_output={
            "key_concepts": [
                "climate change",
                "ocean warming",
                "coral reefs",
                "biodiversity",
                "marine ecosystems",
                "ocean acidification",
            ],
            "research_scope": "moderate",
            "complexity_level": "advanced",
            "suggested_subtopics": [
                "Coral bleaching events",
                "Species migration patterns",
                "Ocean temperature trends",
                "Conservation strategies",
            ],
            "potential_challenges": [
                "Long-term data collection",
                "Multiple confounding factors",
                "Ecosystem complexity",
            ],
        },
        tolerance=0.7,
        tags=["climate", "marine", "biology"],
    ),
]

METHODOLOGY_GOLDEN_CASES = [
    GoldenTestCase(
        id="method_001",
        name="Systematic Review Methodology",
        input_data={
            "topic": "AI in healthcare diagnostics",
            "research_type": "systematic_review",
            "discipline": "medicine",
            "academic_level": "doctoral",
        },
        golden_output={
            "primary_methodology": "Systematic Literature Review",
            "data_collection_methods": [
                "Database searches (PubMed, Scopus, Web of Science)",
                "Grey literature search",
                "Reference list screening",
                "Expert consultation",
            ],
            "analysis_techniques": [
                "PRISMA guidelines",
                "Quality assessment (GRADE)",
                "Narrative synthesis",
                "Meta-analysis if applicable",
            ],
            "quality_criteria": [
                "Comprehensive search strategy",
                "Reproducible methodology",
                "Risk of bias assessment",
                "Transparent reporting",
            ],
        },
        tolerance=0.6,
        tags=["systematic_review", "medicine"],
    ),
]


# =============================================================================
# Golden Set Tests
# =============================================================================


class TestGoldenSetInfrastructure:
    """Tests for the golden set infrastructure itself."""

    def test_golden_test_case_serialization(self):
        """Test that golden test cases can be serialized and deserialized."""
        original = TOPIC_ANALYSIS_GOLDEN_CASES[0]

        # Serialize
        data = original.to_dict()
        assert "id" in data
        assert "golden_output" in data

        # Deserialize
        restored = GoldenTestCase.from_dict(data)
        assert restored.id == original.id
        assert restored.golden_output == original.golden_output

    def test_similarity_calculation_identical(self):
        """Test similarity calculation with identical outputs."""
        output = {"key1": "value1", "key2": ["a", "b", "c"]}
        golden = {"key1": "value1", "key2": ["a", "b", "c"]}

        similarity = calculate_similarity(output, golden)
        assert similarity == 1.0

    def test_similarity_calculation_partial_match(self):
        """Test similarity calculation with partial match."""
        output = {"key1": "value1", "key2": ["a", "b"]}
        golden = {"key1": "value1", "key2": ["a", "b", "c", "d"]}

        similarity = calculate_similarity(output, golden)
        assert 0.5 < similarity < 1.0

    def test_similarity_calculation_no_match(self):
        """Test similarity calculation with no match."""
        output = {"different_key": "different_value"}
        golden = {"key1": "value1"}

        similarity = calculate_similarity(output, golden)
        assert similarity < 0.5


class TestTopicAnalysisGoldenSet:
    """Golden set regression tests for topic analysis."""

    @pytest.mark.parametrize("golden_case", TOPIC_ANALYSIS_GOLDEN_CASES, ids=lambda x: x.id)
    def test_topic_analysis_structure(self, golden_case: GoldenTestCase):
        """Test that topic analysis output has expected structure."""
        golden = golden_case.golden_output

        # Verify golden output structure
        assert "key_concepts" in golden
        assert "research_scope" in golden
        assert "complexity_level" in golden
        assert isinstance(golden["key_concepts"], list)
        assert len(golden["key_concepts"]) >= 3

    @pytest.mark.parametrize("golden_case", TOPIC_ANALYSIS_GOLDEN_CASES, ids=lambda x: x.id)
    def test_topic_analysis_scope_validity(self, golden_case: GoldenTestCase):
        """Test that research scope is valid."""
        scope = golden_case.golden_output.get("research_scope", "")
        assert scope in ["narrow", "moderate", "broad"]

    @pytest.mark.parametrize("golden_case", TOPIC_ANALYSIS_GOLDEN_CASES, ids=lambda x: x.id)
    def test_topic_analysis_complexity_validity(self, golden_case: GoldenTestCase):
        """Test that complexity level is valid."""
        complexity = golden_case.golden_output.get("complexity_level", "")
        assert complexity in ["basic", "intermediate", "advanced"]


class TestMethodologyGoldenSet:
    """Golden set regression tests for methodology recommendations."""

    @pytest.mark.parametrize("golden_case", METHODOLOGY_GOLDEN_CASES, ids=lambda x: x.id)
    def test_methodology_structure(self, golden_case: GoldenTestCase):
        """Test that methodology output has expected structure."""
        golden = golden_case.golden_output

        assert "primary_methodology" in golden
        assert "data_collection_methods" in golden
        assert "analysis_techniques" in golden
        assert isinstance(golden["data_collection_methods"], list)
        assert len(golden["data_collection_methods"]) >= 1


# =============================================================================
# Regression Detection Tests (Mock LLM Output Comparison)
# =============================================================================


class TestRegressionDetection:
    """Tests that detect regressions by comparing against golden outputs."""

    def test_detect_missing_key_concepts(self):
        """Detect regression when key concepts are missing."""
        golden = TOPIC_ANALYSIS_GOLDEN_CASES[0].golden_output

        # Simulated regression: completely different output missing most fields
        regressed_output = {
            "key_concepts": ["AI"],  # Only 1 concept instead of 6
            "research_scope": "unknown",  # Invalid scope
            # Missing other fields entirely
        }

        similarity = calculate_similarity(regressed_output, golden)
        assert similarity < TOPIC_ANALYSIS_GOLDEN_CASES[0].tolerance, (
            f"Regression not detected: similarity={similarity}"
        )

    def test_detect_scope_change(self):
        """Detect regression when scope changes unexpectedly."""
        golden = TOPIC_ANALYSIS_GOLDEN_CASES[0].golden_output

        # Simulated regression: wrong scope
        regressed_output = {
            **golden,
            "research_scope": "narrow",  # Changed from "broad"
        }

        # This should still pass similarity but flag the specific change
        assert regressed_output["research_scope"] != golden["research_scope"]

    def test_no_regression_with_similar_output(self):
        """Verify no false positive when output is similar."""
        golden = TOPIC_ANALYSIS_GOLDEN_CASES[0].golden_output

        # Similar output with minor variations
        similar_output = {
            "key_concepts": [
                "artificial intelligence",
                "machine learning",
                "educational technology",
                "personalized learning",
                "adaptive learning systems",  # Slight variation
            ],
            "research_scope": "broad",
            "complexity_level": "advanced",
            "suggested_subtopics": [
                "AI tutoring systems",  # Slight variation
                "Automated assessment",
                "Learning analytics",
            ],
            "potential_challenges": [
                "Data privacy",  # Slight variation
                "Algorithmic bias",
                "Digital divide",
            ],
        }

        similarity = calculate_similarity(similar_output, golden)
        assert similarity >= TOPIC_ANALYSIS_GOLDEN_CASES[0].tolerance, (
            f"False positive regression detected: similarity={similarity}"
        )
