"""
my_agentic_rag - Agentic RAG Library

Public API:
- get_embedding_service(): Get embedding service (BGE + BM25)
- build_rag_graph(): Build RAG processing graph
"""

from .embedding import get_embedding_service
from .rag_graph import build_rag_graph

__all__ = [
    "get_embedding_service",
    "build_rag_graph",
]
