"""
RAG Service using LangChain Components

Provides a simplified RAG implementation using industry-standard
LangChain libraries for text splitting, embeddings, and vector storage.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

# LangChain imports
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    LANGCHAIN_SPLITTER_AVAILABLE = True
except ImportError:
    LANGCHAIN_SPLITTER_AVAILABLE = False
    logger.warning("langchain-text-splitters not installed")

try:
    from langchain_chroma import Chroma

    LANGCHAIN_CHROMA_AVAILABLE = True
except ImportError:
    LANGCHAIN_CHROMA_AVAILABLE = False
    logger.warning("langchain-chroma not installed")

try:
    from langchain_ollama import OllamaEmbeddings

    LANGCHAIN_OLLAMA_AVAILABLE = True
except ImportError:
    LANGCHAIN_OLLAMA_AVAILABLE = False
    logger.warning("langchain-ollama not installed")

try:
    from langchain_huggingface import HuggingFaceEmbeddings

    LANGCHAIN_HF_AVAILABLE = True
except ImportError:
    LANGCHAIN_HF_AVAILABLE = False
    logger.warning("langchain-huggingface not installed")


@dataclass
class RAGConfig:
    """Configuration for RAG service."""

    # Vector store settings
    persist_directory: str = field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    )

    # Multiple collections for different data types
    collections: Dict[str, str] = field(
        default_factory=lambda: {
            "economic_data": "economic_data",
            "academic_papers": "academic_papers",
            "current_events": "current_events",
            "research_history": "research_history",
        }
    )
    default_collection: str = "research_history"

    # Embedding settings
    embedding_provider: str = "ollama"  # ollama, huggingface
    ollama_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    )
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    hf_model: str = "BAAI/bge-small-en-v1.5"

    # Chunking settings (512 tokens with 50 overlap as per spec)
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Retrieval settings
    retrieval_k: int = 10
    similarity_threshold: float = 0.35  # Lower threshold for better recall

    # Semantic cache settings
    cache_collection_name: str = "semantic_cache"
    cache_similarity_threshold: float = 0.92


class RAGService:
    """
    RAG Service using LangChain components.

    Provides:
    - Document ingestion with automatic chunking
    - Vector storage with ChromaDB
    - Semantic search and retrieval
    - Semantic caching for query responses
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self._embeddings = None
        self._collections: Dict[str, Any] = {}  # Multiple collections
        self._cache_store = None
        self._text_splitter = None

        self._initialize()

    def _initialize(self) -> None:
        """Initialize LangChain components."""
        # Initialize embeddings
        self._embeddings = self._create_embeddings()

        # Initialize text splitter (512 tokens, 50 overlap as per spec)
        if LANGCHAIN_SPLITTER_AVAILABLE:
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "],
            )
            logger.info(
                f"Text splitter initialized: chunk_size={self.config.chunk_size}, "
                f"overlap={self.config.chunk_overlap}"
            )

        # Initialize multiple vector store collections
        if LANGCHAIN_CHROMA_AVAILABLE and self._embeddings:
            # Create collections for each data type
            for collection_key, collection_name in self.config.collections.items():
                self._collections[collection_key] = Chroma(
                    collection_name=collection_name,
                    embedding_function=self._embeddings,
                    persist_directory=self.config.persist_directory,
                )

            # Semantic cache store (Layer 1)
            self._cache_store = Chroma(
                collection_name=self.config.cache_collection_name,
                embedding_function=self._embeddings,
                persist_directory=self.config.persist_directory,
            )

            logger.info(
                f"ChromaDB initialized at: {self.config.persist_directory} "
                f"with collections: {list(self.config.collections.keys())}"
            )

    def _create_embeddings(self) -> Any:
        """Create embedding function based on configuration."""
        if self.config.embedding_provider == "ollama" and LANGCHAIN_OLLAMA_AVAILABLE:
            logger.info(f"Using Ollama embeddings: {self.config.ollama_model}")
            return OllamaEmbeddings(
                model=self.config.ollama_model,
                base_url=self.config.ollama_base_url,
            )
        elif LANGCHAIN_HF_AVAILABLE:
            logger.info(f"Using HuggingFace embeddings: {self.config.hf_model}")
            return HuggingFaceEmbeddings(model_name=self.config.hf_model)
        else:
            logger.error("No embedding provider available")
            return None

    @property
    def is_available(self) -> bool:
        """Check if RAG service is properly initialized."""
        return (
            self._embeddings is not None
            and len(self._collections) > 0
            and self._text_splitter is not None
        )

    def _get_collection(self, collection: Optional[str] = None) -> Any:
        """Get a collection by name or return default."""
        if collection is None:
            collection = self.config.default_collection
        return self._collections.get(
            collection, self._collections.get(self.config.default_collection)
        )

    def ingest_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        collection: Optional[str] = None,
    ) -> List[str]:
        """
        Ingest text into the vector store.

        Args:
            text: Text content to ingest
            metadata: Optional metadata
            collection: Optional collection name (default: main collection)

        Returns:
            List of document IDs
        """
        if not self.is_available:
            logger.warning("RAG service not available")
            return []

        # Split text into chunks
        chunks = self._text_splitter.split_text(text)

        # Create documents with metadata
        base_metadata = metadata or {}
        base_metadata["ingested_at"] = datetime.now().isoformat()

        documents = [
            Document(
                page_content=chunk,
                metadata={**base_metadata, "chunk_index": i, "total_chunks": len(chunks)},
            )
            for i, chunk in enumerate(chunks)
        ]

        # Add to appropriate collection
        store = self._get_collection(collection)
        if store is None:
            logger.warning(f"Collection '{collection}' not found, using default")
            store = self._get_collection()

        ids = store.add_documents(documents)
        logger.debug(
            f"Ingested {len(chunks)} chunks into '{collection or self.config.default_collection}'"
        )
        return ids

    def ingest_documents(
        self,
        documents: List[Dict[str, Any]],
        doc_type: str,
        collection: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Ingest multiple documents into the vector store.

        Args:
            documents: List of document dicts with 'content' and optional 'metadata'
            doc_type: Type of documents (economic_data, academic_papers, etc.)
            collection: Optional collection name

        Returns:
            Statistics dict
        """
        total_chunks = 0
        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            metadata["doc_type"] = doc_type

            ids = self.ingest_text(content, metadata, collection)
            total_chunks += len(ids)

        logger.info(f"Ingested {len(documents)} documents ({total_chunks} chunks) as '{doc_type}'")
        return {"documents": len(documents), "chunks": total_chunks}

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        collection: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            k: Number of results (default: config value)
            filter_dict: Optional metadata filter
            collection: Optional collection name

        Returns:
            List of (Document, score) tuples
        """
        if not self.is_available:
            return []

        k = k or self.config.retrieval_k
        store = self._get_collection(collection)

        # Perform similarity search with scores
        results = store.similarity_search_with_relevance_scores(query, k=k, filter=filter_dict)

        # Filter by similarity threshold
        filtered = [
            (doc, score) for doc, score in results if score >= self.config.similarity_threshold
        ]

        logger.debug(f"Retrieved {len(filtered)}/{len(results)} documents for query")
        return filtered

    def retrieve_as_context(
        self,
        query: str,
        k: Optional[int] = None,
        max_length: int = 8000,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Retrieve documents and format as context for LLM.

        Args:
            query: Search query
            k: Number of results
            max_length: Maximum context length in characters

        Returns:
            Tuple of (formatted_context, source_metadata)
        """
        results = self.retrieve(query, k)

        if not results:
            return "", []

        context_parts = []
        sources = []
        total_length = 0

        for doc, score in results:
            if total_length + len(doc.page_content) > max_length:
                break

            doc_type = doc.metadata.get("doc_type", "unknown")
            context_entry = f"[{doc_type.upper()}]\n{doc.page_content}\n"
            context_parts.append(context_entry)
            total_length += len(context_entry)

            sources.append(
                {
                    "type": doc_type,
                    "score": score,
                    "metadata": doc.metadata,
                }
            )

        formatted_context = "\n---\n".join(context_parts)
        return formatted_context, sources

    # ==================== Semantic Cache Methods ====================

    def cache_lookup(
        self,
        query: str,
        threshold: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Look up a query in the semantic cache.

        Args:
            query: Query to look up
            threshold: Similarity threshold (default: config value)

        Returns:
            Cached response dict if found, None otherwise
        """
        if not self._cache_store:
            return None

        threshold = threshold or self.config.cache_similarity_threshold

        results = self._cache_store.similarity_search_with_relevance_scores(query, k=1)

        if not results:
            return None

        doc, score = results[0]
        if score >= threshold:
            logger.info(f"Semantic cache HIT! Score: {score:.3f}")
            return {
                "query": doc.page_content,
                "response": doc.metadata.get("response", ""),
                "score": score,
                "cached_at": doc.metadata.get("cached_at"),
            }

        return None

    def cache_store(
        self,
        query: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a query-response pair in the semantic cache.

        Args:
            query: Query text
            response: Response text
            metadata: Optional additional metadata

        Returns:
            Document ID
        """
        if not self._cache_store:
            return ""

        cache_metadata = {
            "response": response[:50000],  # Limit size
            "cached_at": datetime.now().isoformat(),
            "query_length": len(query),
            "response_length": len(response),
            **(metadata or {}),
        }

        doc = Document(page_content=query, metadata=cache_metadata)
        ids = self._cache_store.add_documents([doc])

        logger.debug("Cached query-response pair")
        return ids[0] if ids else ""

    async def get_or_compute(
        self,
        query: str,
        compute_fn: Any,
        cache_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, bool]:
        """
        Get from cache or compute and cache the response.

        Args:
            query: Query text
            compute_fn: Async function to compute response
            cache_metadata: Optional metadata for cache entry

        Returns:
            Tuple of (response, was_cached)
        """
        # Check cache first
        cached = self.cache_lookup(query)
        if cached:
            return cached["response"], True

        # Compute response
        response = await compute_fn()

        # Store in cache
        self.cache_store(query, response, cache_metadata)

        return response, False

    # ==================== Utility Methods ====================

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections."""
        stats = {}

        # Stats for each data collection
        for collection_key, store in self._collections.items():
            try:
                stats[collection_key] = {
                    "collection": self.config.collections[collection_key],
                    "count": store._collection.count(),
                }
            except Exception as e:
                stats[collection_key] = {"error": str(e)}

        # Stats for semantic cache
        if self._cache_store:
            try:
                stats["semantic_cache"] = {
                    "collection": self.config.cache_collection_name,
                    "count": self._cache_store._collection.count(),
                }
            except Exception as e:
                stats["semantic_cache"] = {"error": str(e)}

        return stats

    def clear_cache(self) -> int:
        """Clear the semantic cache."""
        if not self._cache_store:
            return 0

        try:
            count = self._cache_store._collection.count()
            self._cache_store._collection.delete(where={"cached_at": {"$exists": True}})
            logger.info(f"Cleared {count} cache entries")
            return count
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0

    def get_retriever(self, k: int = 5):
        """
        Get a LangChain retriever for use in chains.

        Args:
            k: Number of documents to retrieve

        Returns:
            LangChain retriever object
        """
        if not self._vector_store:
            return None

        return self._vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": k,
                "score_threshold": self.config.similarity_threshold,
            },
        )
