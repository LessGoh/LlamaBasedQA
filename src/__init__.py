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
- utils: Utility functions and helpers
- config: Configuration management
"""

__version__ = "1.0.0"
__author__ = "RAG Team"


# Import main components
from . import config
from . import document_processor
from . import metadata_extractor
from . import chunking_simplified
from . import vector_store
from . import query_engine
from . import utils

# Import new workflow components
from . import workflow_events
from . import document_workflow
from . import parallel_manager
from . import parallel_workflow_manager

# Convenience imports for main classes
from .document_processor import ArxivDocumentProcessor, DocumentProcessor
from .metadata_extractor import MetadataExtractor
from .chunking_simplified import HybridChunker
from .vector_store import VectorStore, DocumentIndexer
from .query_engine import QueryEngine
from .utils import QueryCache, ResultFormatter
from .config import settings

# Import new workflow classes
from .workflow_events import (
    DocumentLoadedEvent,
    DocumentParsedEvent,
    MetadataExtractedEvent,
    DocumentChunkedEvent,
    DocumentIndexedEvent,
    DocumentProcessingErrorEvent,
    ProcessingStatus,
    StageStatus
)
from .document_workflow import DocumentProcessingWorkflow, process_single_document
from .parallel_workflow_manager import (
    ParallelWorkflowManager,
    BatchProcessingConfig,
    BatchProcessingResult,
    process_documents_batch
)
from .progress_tracker import (
    AdvancedProgressTracker,
    create_progress_tracker,
    get_default_progress_tracker
)

__all__ = [
    # Modules
    "config",
    "document_processor", 
    "metadata_extractor",
    "chunking",
    "vector_store", 
    "query_engine",
    "utils",
    "workflow_events",
    "document_workflow",
    "parallel_manager",
    "parallel_workflow_manager",
    
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
    "settings",
    
    # Workflow classes
    "DocumentProcessingWorkflow",
    "process_single_document",
    "ParallelWorkflowManager",
    "BatchProcessingConfig",
    "BatchProcessingResult",
    "process_documents_batch",
    "DocumentLoadedEvent",
    "DocumentParsedEvent",
    "MetadataExtractedEvent",
    "DocumentChunkedEvent",
    "DocumentIndexedEvent",
    "DocumentProcessingErrorEvent",
    "ProcessingStatus",
    "StageStatus",
    
    # Progress tracker classes
    "AdvancedProgressTracker",
    "create_progress_tracker",
    "get_default_progress_tracker"
]