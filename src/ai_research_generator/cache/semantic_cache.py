"""
Semantic Cache for Research System

Provides semantic similarity-based caching for LLM responses.
Uses vector embeddings to find similar past queries and return
cached responses.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

from loguru import logger


@dataclass
class SemanticCacheConfig:
    """Configuration for semantic cache."""

    # Similarity settings
    similarity_threshold: float = 0.92  # High threshold for precision

    # Cache settings
    max_entries: int = 500
    ttl_days: int = 7  # Cache entries expire after 7 days

    # Collection name in vector store
    collection_name: str = "semantic_cache"


@dataclass
class CachedResponse:
    """A cached query-response pair."""

    query: str
    response: str
    similarity_score: float
    cached_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    hit_count: int = 0


class SemanticCache:
    """
    Semantic similarity-based cache for LLM responses.

    Uses vector embeddings to find semantically similar queries
    and return cached responses, reducing redundant LLM calls.

    Features:
    - High-precision similarity matching (default threshold: 0.92)
    - TTL-based expiration
    - Metadata tracking for cache analytics
    """

    def __init__(
        self,
        vector_store: Any,
        embedding_service: Any,
        config: Optional[SemanticCacheConfig] = None,
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.config = config or SemanticCacheConfig()

        self._stats = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
        }

        logger.info(
            f"SemanticCache initialized: threshold={self.config.similarity_threshold}, "
            f"ttl={self.config.ttl_days} days"
        )

    async def get(
        self,
        query: str,
        threshold: Optional[float] = None,
    ) -> Optional[CachedResponse]:
        """
        Look up a query in the semantic cache.

        Args:
            query: The query to look up
            threshold: Optional custom similarity threshold

        Returns:
            CachedResponse if found, None otherwise
        """
        threshold = threshold or self.config.similarity_threshold

        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.embed_text(query)

            # Search for similar queries
            results = self.vector_store.query(
                collection_name=self.config.collection_name,
                query_embeddings=[query_embedding],
                n_results=1,
                include=["documents", "metadatas", "distances"],
            )

            if not results or not results.get("documents") or not results["documents"][0]:
                self._stats["misses"] += 1
                return None

            # Check similarity threshold
            distance = results["distances"][0][0]
            # Convert L2 distance to similarity score
            similarity = 1.0 / (1.0 + distance)

            if similarity < threshold:
                self._stats["misses"] += 1
                logger.debug(
                    f"Semantic cache miss: similarity {similarity:.3f} < threshold {threshold}"
                )
                return None

            # Check TTL
            metadata = results["metadatas"][0][0]
            cached_at_str = metadata.get("cached_at")
            if cached_at_str:
                cached_at = datetime.fromisoformat(cached_at_str)
                if datetime.now() - cached_at > timedelta(days=self.config.ttl_days):
                    self._stats["misses"] += 1
                    logger.debug("Semantic cache miss: entry expired")
                    return None

            self._stats["hits"] += 1
            logger.info(f"Semantic cache HIT! Similarity: {similarity:.3f}")

            return CachedResponse(
                query=results["documents"][0][0],
                response=metadata.get("response", ""),
                similarity_score=similarity,
                cached_at=datetime.fromisoformat(cached_at_str)
                if cached_at_str
                else datetime.now(),
                metadata=metadata,
            )

        except Exception as e:
            logger.warning(f"Semantic cache lookup error: {e}")
            self._stats["misses"] += 1
            return None

    async def set(
        self,
        query: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a query-response pair in the semantic cache.

        Args:
            query: The query text
            response: The response text
            metadata: Optional additional metadata

        Returns:
            Cache entry ID
        """
        import uuid

        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.embed_text(query)

            # Prepare metadata
            cache_metadata = {
                "response": response[:50000],  # Limit response size
                "cached_at": datetime.now().isoformat(),
                "query_length": len(query),
                "response_length": len(response),
                **(metadata or {}),
            }

            # Generate unique ID
            cache_id = f"cache_{uuid.uuid4().hex[:12]}"

            # Store in vector store
            self.vector_store.add_documents(
                collection_name=self.config.collection_name,
                documents=[query],
                embeddings=[query_embedding],
                metadatas=[cache_metadata],
                ids=[cache_id],
            )

            self._stats["stores"] += 1
            logger.debug(f"Stored in semantic cache: {cache_id}")

            return cache_id

        except Exception as e:
            logger.error(f"Semantic cache store error: {e}")
            raise

    async def get_or_compute(
        self,
        query: str,
        compute_fn: Any,
        threshold: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, bool]:
        """
        Get from cache or compute and cache the response.

        Args:
            query: The query text
            compute_fn: Async function to compute response if not cached
            threshold: Optional custom similarity threshold
            metadata: Optional metadata for cache entry

        Returns:
            Tuple of (response, was_cached)
        """
        # Try cache first
        cached = await self.get(query, threshold)
        if cached:
            return cached.response, True

        # Compute response
        response = await compute_fn()

        # Store in cache
        await self.set(query, response, metadata)

        return response, False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0

        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "stores": self._stats["stores"],
            "hit_rate": f"{hit_rate:.2%}",
            "total_requests": total,
        }

    async def clear(self) -> int:
        """
        Clear all entries from the semantic cache.

        Returns:
            Number of entries cleared
        """
        try:
            count = self.vector_store.get_collection_count(self.config.collection_name)
            self.vector_store.clear_collection(self.config.collection_name)
            logger.info(f"Cleared {count} entries from semantic cache")
            return count
        except Exception as e:
            logger.error(f"Error clearing semantic cache: {e}")
            return 0

    async def cleanup_expired(self) -> int:
        """
        Remove expired entries from the cache.

        Returns:
            Number of entries removed
        """
        try:
            collection = self.vector_store.get_collection(self.config.collection_name)
            all_data = collection.get(include=["metadatas"])

            if not all_data or not all_data.get("ids"):
                return 0

            expired_ids = []
            cutoff = datetime.now() - timedelta(days=self.config.ttl_days)

            for i, metadata in enumerate(all_data.get("metadatas", [])):
                cached_at_str = metadata.get("cached_at")
                if cached_at_str:
                    cached_at = datetime.fromisoformat(cached_at_str)
                    if cached_at < cutoff:
                        expired_ids.append(all_data["ids"][i])

            if expired_ids:
                self.vector_store.delete_documents(
                    collection_name=self.config.collection_name,
                    ids=expired_ids,
                )
                logger.info(f"Removed {len(expired_ids)} expired cache entries")

            return len(expired_ids)

        except Exception as e:
            logger.error(f"Error cleaning up semantic cache: {e}")
            return 0
