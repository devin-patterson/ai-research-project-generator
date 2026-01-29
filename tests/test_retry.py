"""
Tests for retry utilities.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from app.core.retry import (
    RetryConfig,
    retry_sync,
    retry_async,
    retry_operation,
    DEFAULT_RETRY_CONFIG,
)


class TestRetryConfig:
    """Tests for RetryConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RetryConfig(
            max_retries=5,
            base_delay=0.5,
            max_delay=30.0,
            exponential_base=3.0,
            jitter=False,
        )
        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 3.0
        assert config.jitter is False

    def test_calculate_delay_exponential(self):
        """Test exponential backoff calculation."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)
        
        assert config.calculate_delay(0) == 1.0   # 1 * 2^0 = 1
        assert config.calculate_delay(1) == 2.0   # 1 * 2^1 = 2
        assert config.calculate_delay(2) == 4.0   # 1 * 2^2 = 4
        assert config.calculate_delay(3) == 8.0   # 1 * 2^3 = 8

    def test_calculate_delay_max_cap(self):
        """Test that delay is capped at max_delay."""
        config = RetryConfig(
            base_delay=1.0,
            max_delay=5.0,
            exponential_base=2.0,
            jitter=False,
        )
        
        # 1 * 2^10 = 1024, but should be capped at 5
        assert config.calculate_delay(10) == 5.0

    def test_calculate_delay_with_jitter(self):
        """Test that jitter adds randomness to delay."""
        config = RetryConfig(base_delay=1.0, jitter=True)
        
        # With jitter, delay should be between base and base * 1.25
        delays = [config.calculate_delay(0) for _ in range(100)]
        
        assert all(1.0 <= d <= 1.25 for d in delays)
        # Should have some variation
        assert len(set(delays)) > 1


class TestRetrySyncDecorator:
    """Tests for retry_sync decorator."""

    def test_success_no_retry(self):
        """Test successful call without retry."""
        mock_func = Mock(return_value="success")
        
        @retry_sync(config=RetryConfig(max_retries=3))
        def test_func():
            return mock_func()
        
        result = test_func()
        
        assert result == "success"
        assert mock_func.call_count == 1

    def test_retry_then_success(self):
        """Test retry followed by success."""
        mock_func = Mock(side_effect=[ValueError("fail"), ValueError("fail"), "success"])
        
        config = RetryConfig(
            max_retries=3,
            base_delay=0.01,  # Fast for testing
            retryable_exceptions=(ValueError,),
        )
        
        @retry_sync(config=config)
        def test_func():
            return mock_func()
        
        result = test_func()
        
        assert result == "success"
        assert mock_func.call_count == 3

    def test_all_retries_exhausted(self):
        """Test that exception is raised when all retries exhausted."""
        mock_func = Mock(side_effect=ValueError("always fails"))
        
        config = RetryConfig(
            max_retries=2,
            base_delay=0.01,
            retryable_exceptions=(ValueError,),
        )
        
        @retry_sync(config=config)
        def test_func():
            return mock_func()
        
        with pytest.raises(ValueError, match="always fails"):
            test_func()
        
        # Initial call + 2 retries = 3 calls
        assert mock_func.call_count == 3

    def test_non_retryable_exception(self):
        """Test that non-retryable exceptions are raised immediately."""
        mock_func = Mock(side_effect=TypeError("not retryable"))
        
        config = RetryConfig(
            max_retries=3,
            base_delay=0.01,
            retryable_exceptions=(ValueError,),  # Only ValueError is retryable
        )
        
        @retry_sync(config=config)
        def test_func():
            return mock_func()
        
        with pytest.raises(TypeError, match="not retryable"):
            test_func()
        
        # Should fail immediately without retry
        assert mock_func.call_count == 1

    def test_on_retry_callback(self):
        """Test that on_retry callback is called."""
        mock_func = Mock(side_effect=[ValueError("fail"), "success"])
        on_retry_mock = Mock()
        
        config = RetryConfig(
            max_retries=3,
            base_delay=0.01,
            retryable_exceptions=(ValueError,),
        )
        
        @retry_sync(config=config, on_retry=on_retry_mock)
        def test_func():
            return mock_func()
        
        result = test_func()
        
        assert result == "success"
        assert on_retry_mock.call_count == 1
        # Check callback was called with exception and attempt number
        call_args = on_retry_mock.call_args[0]
        assert isinstance(call_args[0], ValueError)
        assert call_args[1] == 0  # First attempt (0-indexed)


class TestRetryAsyncDecorator:
    """Tests for retry_async decorator."""

    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        """Test successful async call without retry."""
        mock_func = AsyncMock(return_value="success")
        
        @retry_async(config=RetryConfig(max_retries=3))
        async def test_func():
            return await mock_func()
        
        result = await test_func()
        
        assert result == "success"
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_then_success(self):
        """Test async retry followed by success."""
        mock_func = AsyncMock(side_effect=[ValueError("fail"), "success"])
        
        config = RetryConfig(
            max_retries=3,
            base_delay=0.01,
            retryable_exceptions=(ValueError,),
        )
        
        @retry_async(config=config)
        async def test_func():
            return await mock_func()
        
        result = await test_func()
        
        assert result == "success"
        assert mock_func.call_count == 2

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        """Test that exception is raised when all async retries exhausted."""
        mock_func = AsyncMock(side_effect=ValueError("always fails"))
        
        config = RetryConfig(
            max_retries=2,
            base_delay=0.01,
            retryable_exceptions=(ValueError,),
        )
        
        @retry_async(config=config)
        async def test_func():
            return await mock_func()
        
        with pytest.raises(ValueError, match="always fails"):
            await test_func()
        
        assert mock_func.call_count == 3


class TestRetryOperation:
    """Tests for retry_operation function."""

    @pytest.mark.asyncio
    async def test_success(self):
        """Test successful operation."""
        async def operation():
            return "success"
        
        result = await retry_operation(operation)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_then_success(self):
        """Test operation that fails then succeeds."""
        call_count = 0
        
        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("fail")
            return "success"
        
        config = RetryConfig(max_retries=3, base_delay=0.01, retryable_exceptions=(ValueError,))
        result = await retry_operation(operation, config=config)
        
        assert result == "success"
        assert call_count == 3
