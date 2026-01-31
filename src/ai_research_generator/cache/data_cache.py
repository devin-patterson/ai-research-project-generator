"""
TTL-Based Data Cache

Provides in-memory caching with time-to-live (TTL) expiration
for API responses and collected data.
"""

import hashlib
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, TypeVar

from loguru import logger

T = TypeVar("T")


@dataclass
class CacheEntry:
    """A single cache entry with TTL."""

    key: str
    value: Any
    created_at: datetime
    expires_at: datetime
    ttl_seconds: int
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return datetime.now() > self.expires_at

    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return (datetime.now() - self.created_at).total_seconds()


@dataclass
class DataCacheConfig:
    """Configuration for data cache."""

    # Default TTL settings (in seconds)
    default_ttl: int = 3600  # 1 hour

    # TTL by data type
    ttl_economic_data: int = 14400  # 4 hours
    ttl_current_events: int = 3600  # 1 hour
    ttl_academic_papers: int = 86400  # 24 hours
    ttl_web_search: int = 7200  # 2 hours

    # Cache limits
    max_entries: int = 1000
    max_memory_mb: int = 100

    # Cleanup settings
    cleanup_interval_seconds: int = 300  # 5 minutes


class DataCache:
    """
    In-memory cache with TTL expiration.

    Provides:
    - TTL-based expiration
    - Type-specific TTL settings
    - Automatic cleanup of expired entries
    - Thread-safe operations
    - Cache statistics
    """

    def __init__(self, config: Optional[DataCacheConfig] = None):
        self.config = config or DataCacheConfig()
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
        }

        # Start cleanup thread
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()
        self._start_cleanup_thread()

        logger.info(
            f"DataCache initialized: max_entries={self.config.max_entries}, "
            f"default_ttl={self.config.default_ttl}s"
        )

    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name="cache-cleanup"
        )
        self._cleanup_thread.start()

    def _cleanup_loop(self) -> None:
        """Background loop to clean up expired entries."""
        while not self._stop_cleanup.wait(self.config.cleanup_interval_seconds):
            self._cleanup_expired()

    def _cleanup_expired(self) -> int:
        """Remove expired entries from cache."""
        with self._lock:
            expired_keys = [key for key, entry in self._cache.items() if entry.is_expired]
            for key in expired_keys:
                del self._cache[key]
                self._stats["evictions"] += 1

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

            return len(expired_keys)

    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_ttl_for_type(self, data_type: str) -> int:
        """Get TTL for a specific data type."""
        ttl_map = {
            "economic_data": self.config.ttl_economic_data,
            "economic": self.config.ttl_economic_data,
            "current_events": self.config.ttl_current_events,
            "events": self.config.ttl_current_events,
            "academic_papers": self.config.ttl_academic_papers,
            "academic": self.config.ttl_academic_papers,
            "papers": self.config.ttl_academic_papers,
            "web_search": self.config.ttl_web_search,
            "web": self.config.ttl_web_search,
        }
        return ttl_map.get(data_type, self.config.default_ttl)

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats["misses"] += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self._stats["misses"] += 1
                self._stats["evictions"] += 1
                return None

            entry.hit_count += 1
            self._stats["hits"] += 1
            return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        data_type: Optional[str] = None,
    ) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (optional)
            data_type: Data type for automatic TTL selection
        """
        if ttl is None:
            ttl = self._get_ttl_for_type(data_type) if data_type else self.config.default_ttl

        with self._lock:
            # Check cache size limit
            if len(self._cache) >= self.config.max_entries:
                self._evict_oldest()

            now = datetime.now()
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                expires_at=now + timedelta(seconds=ttl),
                ttl_seconds=ttl,
            )
            self._cache[key] = entry
            self._stats["sets"] += 1

    def _evict_oldest(self) -> None:
        """Evict oldest entries when cache is full."""
        if not self._cache:
            return

        # Sort by creation time and remove oldest 10%
        sorted_entries = sorted(self._cache.items(), key=lambda x: x[1].created_at)
        num_to_evict = max(1, len(sorted_entries) // 10)

        for key, _ in sorted_entries[:num_to_evict]:
            del self._cache[key]
            self._stats["evictions"] += 1

        logger.debug(f"Evicted {num_to_evict} oldest cache entries")

    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cleared {count} cache entries")

    def get_or_set(
        self,
        key: str,
        factory: Callable[[], T],
        ttl: Optional[int] = None,
        data_type: Optional[str] = None,
    ) -> T:
        """
        Get from cache or compute and cache value.

        Args:
            key: Cache key
            factory: Function to compute value if not cached
            ttl: Time-to-live in seconds
            data_type: Data type for automatic TTL

        Returns:
            Cached or computed value
        """
        value = self.get(key)
        if value is not None:
            return value

        value = factory()
        self.set(key, value, ttl, data_type)
        return value

    async def get_or_set_async(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: Optional[int] = None,
        data_type: Optional[str] = None,
    ) -> Any:
        """
        Async version of get_or_set.

        Args:
            key: Cache key
            factory: Async function to compute value if not cached
            ttl: Time-to-live in seconds
            data_type: Data type for automatic TTL

        Returns:
            Cached or computed value
        """
        value = self.get(key)
        if value is not None:
            return value

        value = await factory()
        self.set(key, value, ttl, data_type)
        return value

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0

            return {
                "entries": len(self._cache),
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": f"{hit_rate:.2%}",
                "sets": self._stats["sets"],
                "evictions": self._stats["evictions"],
            }

    def get_entry_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get information about a cache entry."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None

            return {
                "key": entry.key,
                "created_at": entry.created_at.isoformat(),
                "expires_at": entry.expires_at.isoformat(),
                "ttl_seconds": entry.ttl_seconds,
                "age_seconds": entry.age_seconds,
                "is_expired": entry.is_expired,
                "hit_count": entry.hit_count,
            }

    def shutdown(self) -> None:
        """Shutdown cache and cleanup thread."""
        self._stop_cleanup.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=1)
        logger.info("DataCache shutdown complete")
