"""
Evaluation Tests using DeepEval.

These tests evaluate LLM output quality using metrics like:
1. Answer Relevancy - Is the response relevant to the input?
2. Hallucination - Does the response contain fabricated information?
3. Faithfulness - Is the response faithful to the provided context?
4. Coherence - Is the response logically coherent?
5. Custom metrics - Domain-specific quality checks
"""

import pytest
from typing import Any

# DeepEval imports (with graceful fallback for CI without API keys)
try:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        HallucinationMetric,
    )
    from deepeval.test_case import LLMTestCase

    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    LLMTestCase = None
    AnswerRelevancyMetric = None
    HallucinationMetric = None

# =============================================================================
# Custom Evaluation Metrics
# =============================================================================


class ResearchQualityMetric:
    """Custom metric for evaluating research output quality."""

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: dict) -> float:
        """
        Measure research quality based on multiple factors.

        Factors:
        - Key concepts coverage (0-25 points)
        - Methodology appropriateness (0-25 points)
        - Source quality (0-25 points)
        - Coherence and structure (0-25 points)
        """
        output = test_case.get("actual_output", {})

        score = 0.0
        reasons = []

        # Key concepts coverage
        if "key_concepts" in output:
            concepts = output["key_concepts"]
            if isinstance(concepts, list) and len(concepts) >= 5:
                score += 25
                reasons.append("Good key concept coverage")
            elif isinstance(concepts, list) and len(concepts) >= 3:
                score += 15
                reasons.append("Adequate key concept coverage")
            else:
                reasons.append("Insufficient key concepts")

        # Methodology appropriateness
        if "methodology_recommendations" in output:
            methods = output["methodology_recommendations"]
            if isinstance(methods, list) and len(methods) >= 3:
                score += 25
                reasons.append("Comprehensive methodology")
            elif isinstance(methods, list) and len(methods) >= 1:
                score += 15
                reasons.append("Basic methodology provided")
            else:
                reasons.append("Missing methodology")

        # Source quality (papers)
        if "papers" in output:
            papers = output["papers"]
            if isinstance(papers, list) and len(papers) >= 10:
                score += 25
                reasons.append("Excellent source coverage")
            elif isinstance(papers, list) and len(papers) >= 5:
                score += 15
                reasons.append("Adequate sources")
            else:
                reasons.append("Limited sources")

        # Coherence (check for required fields)
        required_fields = ["topic", "research_question", "quality_score"]
        present_fields = sum(1 for f in required_fields if f in output)
        coherence_score = (present_fields / len(required_fields)) * 25
        score += coherence_score
        if coherence_score >= 20:
            reasons.append("Good structural coherence")

        self.score = score / 100.0
        self.reason = "; ".join(reasons)

        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold


class TopicRelevanceMetric:
    """Metric for evaluating topic relevance of LLM outputs."""

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: dict) -> float:
        """
        Measure how relevant the output is to the input topic.

        Uses keyword overlap and semantic similarity approximation.
        """
        input_text = test_case.get("input", "").lower()
        output = test_case.get("actual_output", {})

        # Extract words from input
        input_words = set(input_text.split())
        stop_words = {
            "the",
            "a",
            "an",
            "of",
            "in",
            "on",
            "for",
            "to",
            "and",
            "or",
            "is",
            "are",
            "how",
            "does",
            "what",
        }
        input_keywords = input_words - stop_words

        # Extract words from output
        output_text = ""
        if isinstance(output, dict):
            for value in output.values():
                if isinstance(value, str):
                    output_text += " " + value
                elif isinstance(value, list):
                    output_text += " " + " ".join(str(v) for v in value)
        elif isinstance(output, str):
            output_text = output

        output_words = set(output_text.lower().split())

        # Calculate overlap
        if not input_keywords:
            self.score = 0.5  # Neutral if no keywords
            self.reason = "No input keywords to compare"
            return self.score

        overlap = len(input_keywords & output_words)
        self.score = min(1.0, overlap / len(input_keywords))
        self.reason = f"Keyword overlap: {overlap}/{len(input_keywords)}"

        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_research_output() -> dict[str, Any]:
    """Sample research output for evaluation."""
    return {
        "topic": "Impact of artificial intelligence on education",
        "research_question": "How does AI affect student learning outcomes?",
        "key_concepts": [
            "artificial intelligence",
            "machine learning",
            "educational technology",
            "personalized learning",
            "adaptive systems",
            "student outcomes",
        ],
        "methodology_recommendations": [
            "Systematic literature review",
            "Meta-analysis of existing studies",
            "Survey of educators and students",
            "Longitudinal outcome tracking",
        ],
        "papers": [{"title": f"Paper {i}", "year": 2023} for i in range(12)],
        "quality_score": 85.0,
    }


@pytest.fixture
def sample_poor_output() -> dict[str, Any]:
    """Sample poor quality output for evaluation."""
    return {
        "topic": "AI in education",
        "key_concepts": ["AI"],  # Too few
        # Missing methodology_recommendations
        "papers": [],  # No papers
        # Missing quality_score
    }


# =============================================================================
# Custom Metric Tests
# =============================================================================


class TestResearchQualityMetric:
    """Tests for the custom ResearchQualityMetric."""

    def test_high_quality_output(self, sample_research_output):
        """Test that high quality output scores well."""
        metric = ResearchQualityMetric(threshold=0.7)

        test_case = {
            "actual_output": sample_research_output,
            "expected_output": {},
        }

        score = metric.measure(test_case)

        assert score >= 0.7, f"High quality output should score >= 0.7, got {score}"
        assert metric.is_successful()

    def test_poor_quality_output(self, sample_poor_output):
        """Test that poor quality output scores low."""
        metric = ResearchQualityMetric(threshold=0.7)

        test_case = {
            "actual_output": sample_poor_output,
            "expected_output": {},
        }

        score = metric.measure(test_case)

        assert score < 0.5, f"Poor quality output should score < 0.5, got {score}"
        assert not metric.is_successful()

    def test_metric_provides_reasons(self, sample_research_output):
        """Test that metric provides explanatory reasons."""
        metric = ResearchQualityMetric()

        test_case = {"actual_output": sample_research_output}
        metric.measure(test_case)

        assert metric.reason != ""
        assert "concept" in metric.reason.lower() or "methodology" in metric.reason.lower()


class TestTopicRelevanceMetric:
    """Tests for the TopicRelevanceMetric."""

    def test_relevant_output(self, sample_research_output):
        """Test that relevant output scores well."""
        metric = TopicRelevanceMetric(threshold=0.5)

        test_case = {
            "input": "Impact of artificial intelligence on education",
            "actual_output": sample_research_output,
        }

        score = metric.measure(test_case)

        assert score >= 0.5, f"Relevant output should score >= 0.5, got {score}"

    def test_irrelevant_output(self):
        """Test that irrelevant output scores low."""
        metric = TopicRelevanceMetric(threshold=0.5)

        test_case = {
            "input": "Impact of artificial intelligence on education",
            "actual_output": {
                "topic": "Cooking recipes",
                "key_concepts": ["pasta", "sauce", "ingredients"],
            },
        }

        score = metric.measure(test_case)

        assert score < 0.3, f"Irrelevant output should score < 0.3, got {score}"


# =============================================================================
# DeepEval Integration Tests (Skipped if DeepEval not available)
# =============================================================================


@pytest.mark.skipif(not DEEPEVAL_AVAILABLE, reason="DeepEval not installed")
class TestDeepEvalIntegration:
    """Integration tests using DeepEval metrics.

    Note: These tests require OPENAI_API_KEY to be set for actual DeepEval metrics.
    We skip them in CI environments without API keys.
    """

    @pytest.mark.skip(reason="Requires OPENAI_API_KEY - run manually with API key set")
    def test_answer_relevancy_metric(self):
        """Test answer relevancy using DeepEval."""
        test_case = LLMTestCase(
            input="What are the key benefits of AI in education?",
            actual_output="AI in education provides personalized learning, "
            "automated assessment, and adaptive content delivery.",
        )

        metric = AnswerRelevancyMetric(threshold=0.7)
        metric.measure(test_case)

        assert metric.score >= 0.7

    @pytest.mark.skip(reason="Requires OPENAI_API_KEY - run manually with API key set")
    def test_hallucination_metric(self):
        """Test hallucination detection using DeepEval."""
        test_case = LLMTestCase(
            input="What is the capital of France?",
            actual_output="The capital of France is Paris.",
            context=["France is a country in Europe.", "Paris is the capital of France."],
        )

        metric = HallucinationMetric(threshold=0.5)
        metric.measure(test_case)

        # Low hallucination score is good
        assert metric.score <= 0.5


# =============================================================================
# Evaluation Suite Tests
# =============================================================================


class TestEvaluationSuite:
    """Comprehensive evaluation suite for research outputs."""

    def test_full_evaluation_pipeline(self, sample_research_output):
        """Run full evaluation pipeline on sample output."""
        metrics = [
            ResearchQualityMetric(threshold=0.6),
            TopicRelevanceMetric(threshold=0.4),
        ]

        test_case = {
            "input": "Impact of artificial intelligence on education",
            "actual_output": sample_research_output,
            "expected_output": {},
        }

        results = []
        for metric in metrics:
            score = metric.measure(test_case)
            results.append(
                {
                    "metric": metric.__class__.__name__,
                    "score": score,
                    "passed": metric.is_successful(),
                    "reason": metric.reason,
                }
            )

        # All metrics should pass for good output
        assert all(r["passed"] for r in results), f"Failed metrics: {results}"

    def test_evaluation_report_generation(self, sample_research_output):
        """Test that evaluation generates a proper report."""
        metrics = [
            ResearchQualityMetric(threshold=0.6),
            TopicRelevanceMetric(threshold=0.4),
        ]

        test_case = {
            "input": "AI in education",
            "actual_output": sample_research_output,
        }

        report = {
            "test_case_id": "eval_001",
            "input": test_case["input"],
            "metrics": [],
            "overall_pass": True,
        }

        for metric in metrics:
            score = metric.measure(test_case)
            report["metrics"].append(
                {
                    "name": metric.__class__.__name__,
                    "score": round(score, 3),
                    "threshold": metric.threshold,
                    "passed": metric.is_successful(),
                }
            )
            if not metric.is_successful():
                report["overall_pass"] = False

        # Verify report structure
        assert "test_case_id" in report
        assert "metrics" in report
        assert len(report["metrics"]) == 2
        assert all("score" in m for m in report["metrics"])


# =============================================================================
# Batch Evaluation Tests
# =============================================================================


class TestBatchEvaluation:
    """Tests for batch evaluation of multiple outputs."""

    def test_batch_evaluation(self):
        """Test evaluating multiple outputs in batch."""
        test_cases = [
            {
                "id": "batch_001",
                "input": "AI in healthcare",
                "actual_output": {
                    "topic": "AI in healthcare",
                    "key_concepts": ["AI", "healthcare", "diagnostics", "treatment", "prediction"],
                    "methodology_recommendations": ["systematic review", "meta-analysis"],
                    "papers": [{"title": f"Paper {i}"} for i in range(8)],
                    "quality_score": 80,
                },
            },
            {
                "id": "batch_002",
                "input": "Climate change impacts",
                "actual_output": {
                    "topic": "Climate change",
                    "key_concepts": ["climate", "warming", "emissions"],
                    "methodology_recommendations": ["literature review"],
                    "papers": [{"title": f"Paper {i}"} for i in range(3)],
                    "quality_score": 60,
                },
            },
        ]

        metric = ResearchQualityMetric(threshold=0.5)

        results = []
        for tc in test_cases:
            score = metric.measure(tc)
            results.append(
                {
                    "id": tc["id"],
                    "score": score,
                    "passed": metric.is_successful(),
                }
            )

        # At least one should pass
        assert any(r["passed"] for r in results)

        # Calculate aggregate stats
        avg_score = sum(r["score"] for r in results) / len(results)
        pass_rate = sum(1 for r in results if r["passed"]) / len(results)

        assert avg_score > 0.3
        assert pass_rate >= 0.5
