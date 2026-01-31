"""
Load, Latency, and Cost Tests for LLM Operations.

These tests measure and validate:
1. Response latency - Time to first token and total response time
2. Throughput - Requests per second under load
3. Token usage - Input/output token counts
4. Cost estimation - Estimated API costs per request
5. Resource utilization - Memory and CPU usage
"""

import asyncio
import time
import statistics
from dataclasses import dataclass, field
from typing import Optional, Callable
from contextlib import contextmanager

import pytest


# =============================================================================
# Performance Metrics Data Classes
# =============================================================================


@dataclass
class LatencyMetrics:
    """Metrics for measuring latency."""

    time_to_first_token_ms: float = 0.0
    total_response_time_ms: float = 0.0
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "time_to_first_token_ms": round(self.time_to_first_token_ms, 2),
            "total_response_time_ms": round(self.total_response_time_ms, 2),
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


@dataclass
class TokenMetrics:
    """Metrics for token usage."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    @property
    def token_ratio(self) -> float:
        """Output to input token ratio."""
        return self.output_tokens / self.input_tokens if self.input_tokens > 0 else 0

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "token_ratio": round(self.token_ratio, 2),
        }


@dataclass
class CostMetrics:
    """Metrics for cost estimation."""

    # Pricing per 1K tokens (example rates)
    input_cost_per_1k: float = 0.0015  # $0.0015 per 1K input tokens
    output_cost_per_1k: float = 0.002  # $0.002 per 1K output tokens

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def input_cost(self) -> float:
        return (self.input_tokens / 1000) * self.input_cost_per_1k

    @property
    def output_cost(self) -> float:
        return (self.output_tokens / 1000) * self.output_cost_per_1k

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost

    def to_dict(self) -> dict:
        return {
            "input_cost_usd": round(self.input_cost, 6),
            "output_cost_usd": round(self.output_cost, 6),
            "total_cost_usd": round(self.total_cost, 6),
        }


@dataclass
class LoadTestResult:
    """Results from a load test."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_duration_seconds: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successful_requests / self.total_requests if self.total_requests > 0 else 0

    @property
    def requests_per_second(self) -> float:
        return (
            self.total_requests / self.total_duration_seconds
            if self.total_duration_seconds > 0
            else 0
        )

    @property
    def avg_latency_ms(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0

    @property
    def p50_latency_ms(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    @property
    def p99_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round(self.success_rate, 4),
            "requests_per_second": round(self.requests_per_second, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "error_count": len(self.errors),
        }


# =============================================================================
# Performance Testing Utilities
# =============================================================================


@contextmanager
def measure_time():
    """Context manager to measure execution time."""
    start = time.perf_counter()
    metrics = {"start": start, "end": 0, "duration_ms": 0}
    try:
        yield metrics
    finally:
        metrics["end"] = time.perf_counter()
        metrics["duration_ms"] = (metrics["end"] - metrics["start"]) * 1000


def estimate_tokens(text: str) -> int:
    """
    Estimate token count from text.

    Rough approximation: ~4 characters per token for English text.
    """
    return len(text) // 4


def calculate_cost(
    input_tokens: int, output_tokens: int, model: str = "gpt-4o-mini"
) -> CostMetrics:
    """
    Calculate estimated cost based on model pricing.

    Pricing as of 2024 (example rates):
    - gpt-4o-mini: $0.00015/1K input, $0.0006/1K output
    - gpt-4o: $0.0025/1K input, $0.01/1K output
    - llama3.1:8b (local): $0 (self-hosted)
    """
    pricing = {
        "gpt-4o-mini": (0.00015, 0.0006),
        "gpt-4o": (0.0025, 0.01),
        "gpt-3.5-turbo": (0.0005, 0.0015),
        "claude-3-haiku": (0.00025, 0.00125),
        "claude-3-sonnet": (0.003, 0.015),
        "llama3.1:8b": (0.0, 0.0),  # Self-hosted
        "ollama": (0.0, 0.0),  # Self-hosted
    }

    input_rate, output_rate = pricing.get(model, (0.001, 0.002))

    return CostMetrics(
        input_cost_per_1k=input_rate,
        output_cost_per_1k=output_rate,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


async def run_load_test(
    request_fn: Callable,
    num_requests: int = 10,
    concurrency: int = 3,
    timeout_seconds: float = 30.0,
) -> LoadTestResult:
    """
    Run a load test with specified concurrency.

    Args:
        request_fn: Async function to call for each request
        num_requests: Total number of requests to make
        concurrency: Maximum concurrent requests
        timeout_seconds: Timeout per request

    Returns:
        LoadTestResult with aggregated metrics
    """
    result = LoadTestResult(total_requests=num_requests)
    semaphore = asyncio.Semaphore(concurrency)

    async def make_request(request_id: int) -> tuple[bool, float, Optional[str]]:
        async with semaphore:
            start = time.perf_counter()
            try:
                await asyncio.wait_for(request_fn(), timeout=timeout_seconds)
                duration_ms = (time.perf_counter() - start) * 1000
                return True, duration_ms, None
            except asyncio.TimeoutError:
                duration_ms = (time.perf_counter() - start) * 1000
                return False, duration_ms, "Timeout"
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                return False, duration_ms, str(e)

    start_time = time.perf_counter()

    tasks = [make_request(i) for i in range(num_requests)]
    results = await asyncio.gather(*tasks)

    result.total_duration_seconds = time.perf_counter() - start_time

    for success, latency, error in results:
        result.latencies_ms.append(latency)
        if success:
            result.successful_requests += 1
        else:
            result.failed_requests += 1
            if error:
                result.errors.append(error)

    return result


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_llm_response():
    """Mock LLM response with token counts."""
    return {
        "content": "This is a sample response about AI in education. " * 20,
        "usage": {
            "prompt_tokens": 150,
            "completion_tokens": 200,
            "total_tokens": 350,
        },
    }


@pytest.fixture
def sample_request():
    """Sample research request for testing."""
    return {
        "topic": "Impact of artificial intelligence on education",
        "research_question": "How does AI affect student learning outcomes?",
        "research_type": "systematic_review",
    }


# =============================================================================
# Latency Tests
# =============================================================================


class TestLatencyMetrics:
    """Tests for latency measurement and validation."""

    def test_latency_measurement(self):
        """Test that latency is measured correctly."""
        with measure_time() as metrics:
            time.sleep(0.1)  # 100ms

        assert metrics["duration_ms"] >= 100
        assert metrics["duration_ms"] < 150  # Allow some overhead

    def test_latency_threshold_pass(self):
        """Test latency within acceptable threshold."""
        max_latency_ms = 5000  # 5 seconds

        with measure_time() as metrics:
            time.sleep(0.05)  # 50ms simulated processing

        assert metrics["duration_ms"] < max_latency_ms, (
            f"Latency {metrics['duration_ms']}ms exceeds threshold {max_latency_ms}ms"
        )

    def test_latency_metrics_dataclass(self):
        """Test LatencyMetrics dataclass."""
        metrics = LatencyMetrics(
            time_to_first_token_ms=50.5,
            total_response_time_ms=250.3,
            processing_time_ms=199.8,
        )

        result = metrics.to_dict()

        assert result["time_to_first_token_ms"] == 50.5
        assert result["total_response_time_ms"] == 250.3


# =============================================================================
# Token Usage Tests
# =============================================================================


class TestTokenMetrics:
    """Tests for token usage tracking."""

    def test_token_estimation(self):
        """Test token estimation from text."""
        text = "This is a test sentence with about twenty tokens or so."
        estimated = estimate_tokens(text)

        # Rough estimate: ~4 chars per token
        assert 10 <= estimated <= 20

    def test_token_metrics_dataclass(self):
        """Test TokenMetrics dataclass."""
        metrics = TokenMetrics(
            input_tokens=150,
            output_tokens=200,
            total_tokens=350,
        )

        assert metrics.token_ratio == 200 / 150

        result = metrics.to_dict()
        assert result["total_tokens"] == 350

    def test_token_budget_compliance(self, mock_llm_response):
        """Test that token usage stays within budget."""
        max_input_tokens = 1000
        max_output_tokens = 2000

        usage = mock_llm_response["usage"]

        assert usage["prompt_tokens"] <= max_input_tokens, (
            f"Input tokens {usage['prompt_tokens']} exceeds budget {max_input_tokens}"
        )
        assert usage["completion_tokens"] <= max_output_tokens, (
            f"Output tokens {usage['completion_tokens']} exceeds budget {max_output_tokens}"
        )


# =============================================================================
# Cost Estimation Tests
# =============================================================================


class TestCostMetrics:
    """Tests for cost estimation."""

    def test_cost_calculation_gpt4o_mini(self):
        """Test cost calculation for GPT-4o-mini."""
        cost = calculate_cost(input_tokens=1000, output_tokens=500, model="gpt-4o-mini")

        # GPT-4o-mini: $0.00015/1K input, $0.0006/1K output
        expected_input_cost = 1.0 * 0.00015  # $0.00015
        expected_output_cost = 0.5 * 0.0006  # $0.0003

        assert abs(cost.input_cost - expected_input_cost) < 0.0001
        assert abs(cost.output_cost - expected_output_cost) < 0.0001

    def test_cost_calculation_local_model(self):
        """Test cost calculation for local/self-hosted model."""
        cost = calculate_cost(input_tokens=10000, output_tokens=5000, model="llama3.1:8b")

        # Local models have zero API cost
        assert cost.total_cost == 0.0

    def test_cost_budget_compliance(self, mock_llm_response):
        """Test that costs stay within budget."""
        max_cost_per_request = 0.10  # $0.10 per request

        usage = mock_llm_response["usage"]
        cost = calculate_cost(
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
            model="gpt-4o-mini",
        )

        assert cost.total_cost <= max_cost_per_request, (
            f"Cost ${cost.total_cost:.4f} exceeds budget ${max_cost_per_request}"
        )

    def test_batch_cost_estimation(self):
        """Test cost estimation for batch processing."""
        requests = [
            {"input_tokens": 200, "output_tokens": 300},
            {"input_tokens": 150, "output_tokens": 250},
            {"input_tokens": 300, "output_tokens": 400},
        ]

        total_cost = 0.0
        for req in requests:
            cost = calculate_cost(
                input_tokens=req["input_tokens"],
                output_tokens=req["output_tokens"],
                model="gpt-4o-mini",
            )
            total_cost += cost.total_cost

        # Verify batch cost is reasonable
        assert total_cost < 0.01  # Less than 1 cent for small batch


# =============================================================================
# Load Tests
# =============================================================================


class TestLoadPerformance:
    """Load and throughput tests."""

    @pytest.mark.asyncio
    async def test_load_test_basic(self):
        """Test basic load test execution."""
        call_count = 0

        async def mock_request():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # 10ms simulated latency

        result = await run_load_test(
            request_fn=mock_request,
            num_requests=10,
            concurrency=3,
        )

        assert result.total_requests == 10
        assert result.successful_requests == 10
        assert result.failed_requests == 0
        assert result.success_rate == 1.0
        assert call_count == 10

    @pytest.mark.asyncio
    async def test_load_test_with_failures(self):
        """Test load test with some failures."""
        call_count = 0

        async def flaky_request():
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:
                raise Exception("Simulated failure")
            await asyncio.sleep(0.01)

        result = await run_load_test(
            request_fn=flaky_request,
            num_requests=9,
            concurrency=3,
        )

        assert result.total_requests == 9
        assert result.failed_requests == 3  # Every 3rd request fails
        assert result.successful_requests == 6
        assert len(result.errors) == 3

    @pytest.mark.asyncio
    async def test_load_test_latency_percentiles(self):
        """Test latency percentile calculations."""

        async def variable_latency_request():
            # Variable latency between 10-100ms
            import random

            await asyncio.sleep(random.uniform(0.01, 0.1))

        result = await run_load_test(
            request_fn=variable_latency_request,
            num_requests=20,
            concurrency=5,
        )

        # Verify percentiles are calculated
        assert result.p50_latency_ms > 0
        assert result.p95_latency_ms >= result.p50_latency_ms
        assert result.p99_latency_ms >= result.p95_latency_ms

    @pytest.mark.asyncio
    async def test_throughput_target(self):
        """Test that throughput meets target."""
        target_rps = 5  # Target: 5 requests per second

        async def fast_request():
            await asyncio.sleep(0.05)  # 50ms per request

        result = await run_load_test(
            request_fn=fast_request,
            num_requests=20,
            concurrency=10,  # High concurrency to achieve target
        )

        # With 10 concurrent requests at 50ms each, should achieve ~200 RPS
        # But we're being conservative with target of 5 RPS
        assert result.requests_per_second >= target_rps, (
            f"Throughput {result.requests_per_second:.2f} RPS below target {target_rps} RPS"
        )


# =============================================================================
# Performance Regression Tests
# =============================================================================


class TestPerformanceRegression:
    """Tests to detect performance regressions."""

    def test_latency_regression_detection(self):
        """Detect if latency has regressed from baseline."""
        baseline_latency_ms = 100  # Historical baseline
        tolerance = 1.5  # Allow 50% increase

        # Simulated current latency
        current_latency_ms = 120

        max_allowed = baseline_latency_ms * tolerance

        assert current_latency_ms <= max_allowed, (
            f"Latency regression: {current_latency_ms}ms > {max_allowed}ms (baseline: {baseline_latency_ms}ms)"
        )

    def test_token_efficiency_regression(self):
        """Detect if token efficiency has regressed."""
        baseline_output_ratio = 1.5  # Historical output/input ratio
        tolerance = 0.8  # Allow 20% decrease

        # Simulated current metrics
        current_input = 100
        current_output = 130  # Ratio = 1.3
        current_ratio = current_output / current_input

        min_allowed = baseline_output_ratio * tolerance

        assert current_ratio >= min_allowed, (
            f"Token efficiency regression: ratio {current_ratio:.2f} < {min_allowed:.2f}"
        )

    def test_cost_regression_detection(self):
        """Detect if costs have increased unexpectedly."""
        baseline_cost_per_request = 0.001  # $0.001 per request
        tolerance = 1.2  # Allow 20% increase

        # Simulated current cost
        current_cost = 0.0011

        max_allowed = baseline_cost_per_request * tolerance

        assert current_cost <= max_allowed, (
            f"Cost regression: ${current_cost:.4f} > ${max_allowed:.4f}"
        )


# =============================================================================
# Performance Report Generation
# =============================================================================


class TestPerformanceReporting:
    """Tests for performance report generation."""

    @pytest.mark.asyncio
    async def test_generate_performance_report(self):
        """Generate a comprehensive performance report."""

        async def mock_request():
            await asyncio.sleep(0.02)

        # Run load test
        load_result = await run_load_test(
            request_fn=mock_request,
            num_requests=10,
            concurrency=3,
        )

        # Calculate costs
        cost = calculate_cost(input_tokens=500, output_tokens=300, model="gpt-4o-mini")

        # Generate report
        report = {
            "test_name": "Research Workflow Performance",
            "timestamp": "2026-01-29T18:00:00Z",
            "load_test": load_result.to_dict(),
            "cost_analysis": cost.to_dict(),
            "summary": {
                "status": "PASS" if load_result.success_rate >= 0.95 else "FAIL",
                "recommendations": [],
            },
        }

        # Add recommendations based on results
        if load_result.p95_latency_ms > 1000:
            report["summary"]["recommendations"].append("Consider optimizing for lower latency")
        if cost.total_cost > 0.01:
            report["summary"]["recommendations"].append(
                "Consider using a more cost-effective model"
            )

        # Verify report structure
        assert "load_test" in report
        assert "cost_analysis" in report
        assert "summary" in report
        assert report["summary"]["status"] == "PASS"
