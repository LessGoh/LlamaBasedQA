"""
RAG System for ArXiv Scientific Publications Analysis

This package provides a complete RAG (Retrieval-Augmented Generation) system
for analyzing ArXiv scientific publications in economics, quantum finance, 
and machine learning.

Main components:
- document_processor: PDF processing with LlamaParse
- metadata_extractor: Structured metadata extraction with LlamaExtract
- chunking: Hybrid chunking strategy for different content types
- vector_store: Pinecone integration with OpenAI embeddings
- query_engine: Query processing with Cohere reranking and GPT-4o-mini generation
- logging_config: Comprehensive logging and monitoring
- utils: Utility functions and helpers
- config: Configuration management
"""

__version__ = "1.0.0"
__author__ = "RAG Team"

# Initialize logging first
from . import logging_config

# Import main components
from . import config
from . import document_processor
from . import metadata_extractor
from . import chunking
from . import vector_store
from . import query_engine
from . import utils

# Convenience imports for main classes
from .document_processor import ArxivDocumentProcessor, DocumentProcessor
from .metadata_extractor import MetadataExtractor
from .chunking import HybridChunker
from .vector_store import VectorStore, DocumentIndexer
from .query_engine import QueryEngine
from .utils import QueryCache, ResultFormatter
from .config import settings

__all__ = [
    # Modules
    "config",
    "document_processor", 
    "metadata_extractor",
    "chunking",
    "vector_store", 
    "query_engine",
    "utils",
    "logging_config",
    
    # Main classes
    "ArxivDocumentProcessor",
    "DocumentProcessor", 
    "MetadataExtractor",
    "HybridChunker",
    "VectorStore",
    "DocumentIndexer", 
    "QueryEngine",
    "QueryCache",
    "ResultFormatter",
    "settings"
]