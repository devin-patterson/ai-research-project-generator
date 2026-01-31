"""
Retry utilities with exponential backoff.

Provides decorators and utilities for retrying failed operations
with configurable backoff strategies.
"""

import asyncio
import random
from functools import wraps
from typing import Callable, Type, Tuple, Optional, Any

from loguru import logger


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    ):
        """
        Initialize retry configuration.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            exponential_base: Base for exponential backoff calculation
            jitter: Whether to add random jitter to delays
            retryable_exceptions: Tuple of exception types that trigger retry
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for a given attempt number.

        Uses exponential backoff with optional jitter.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        delay = self.base_delay * (self.exponential_base**attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            # Add random jitter between 0% and 25% of delay
            jitter_amount = delay * random.uniform(0, 0.25)  # nosec: B311
            delay += jitter_amount

        return delay


# Default configurations for different use cases
DEFAULT_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
)

LLM_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    base_delay=2.0,
    max_delay=60.0,
    exponential_base=2.0,
)

ACADEMIC_SEARCH_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0,
)


def retry_sync(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable:
    """
    Decorator for synchronous functions with retry logic.

    Args:
        config: Retry configuration (uses DEFAULT_RETRY_CONFIG if None)
        on_retry: Optional callback called on each retry with (exception, attempt)

    Returns:
        Decorated function with retry logic
    """
    if config is None:
        config = DEFAULT_RETRY_CONFIG

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt < config.max_retries:
                        delay = config.calculate_delay(attempt)
                        logger.warning(
                            f"Retry {attempt + 1}/{config.max_retries} for {func.__name__} "
                            f"after {delay:.2f}s due to: {e}"
                        )

                        if on_retry:
                            on_retry(e, attempt)

                        import time

                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {config.max_retries} retries exhausted for {func.__name__}: {e}"
                        )

            raise last_exception

        return wrapper

    return decorator


def retry_async(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable:
    """
    Decorator for async functions with retry logic.

    Args:
        config: Retry configuration (uses DEFAULT_RETRY_CONFIG if None)
        on_retry: Optional callback called on each retry with (exception, attempt)

    Returns:
        Decorated async function with retry logic
    """
    if config is None:
        config = DEFAULT_RETRY_CONFIG

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None

            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt < config.max_retries:
                        delay = config.calculate_delay(attempt)
                        logger.warning(
                            f"Retry {attempt + 1}/{config.max_retries} for {func.__name__} "
                            f"after {delay:.2f}s due to: {e}"
                        )

                        if on_retry:
                            on_retry(e, attempt)

                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"All {config.max_retries} retries exhausted for {func.__name__}: {e}"
                        )

            raise last_exception

        return wrapper

    return decorator


async def retry_operation(
    operation: Callable,
    config: Optional[RetryConfig] = None,
    operation_name: str = "operation",
) -> Any:
    """
    Execute an async operation with retry logic.

    Useful for one-off retries without decorating a function.

    Args:
        operation: Async callable to execute
        config: Retry configuration
        operation_name: Name for logging purposes

    Returns:
        Result of the operation

    Raises:
        Last exception if all retries exhausted
    """
    if config is None:
        config = DEFAULT_RETRY_CONFIG

    last_exception: Optional[Exception] = None

    for attempt in range(config.max_retries + 1):
        try:
            return await operation()
        except config.retryable_exceptions as e:
            last_exception = e

            if attempt < config.max_retries:
                delay = config.calculate_delay(attempt)
                logger.warning(
                    f"Retry {attempt + 1}/{config.max_retries} for {operation_name} "
                    f"after {delay:.2f}s due to: {e}"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"All {config.max_retries} retries exhausted for {operation_name}: {e}"
                )

    raise last_exception
