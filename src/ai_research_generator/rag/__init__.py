"""
RAG (Retrieval Augmented Generation) Module

Provides vector storage, embeddings, document ingestion, and retrieval
capabilities for the research system.

Uses LangChain components for industry-standard implementation:
- langchain-chroma: Vector storage with ChromaDB
- langchain-ollama: Local embeddings via Ollama
- langchain-huggingface: HuggingFace embeddings fallback
- langchain-text-splitters: Document chunking
"""

from .rag_service import RAGService, RAGConfig

__all__ = [
    "RAGService",
    "RAGConfig",
]
