"""
Workflow Events Module for ArXiv RAG System

This module defines all the typed events used in the LlamaIndex Workflow system
for parallel document processing. Each event represents a stage in the document
processing pipeline and carries the necessary data for the next stage.

Based on the plan for creating a parallel document processing system with queue.
"""

import logging
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from pydantic import BaseModel, Field, validator
from llama_index.core.workflow import Event
from llama_index.core import Document
from llama_index.core.schema import TextNode
from enum import Enum

# Set up logger
logger = logging.getLogger(__name__)


class ProcessingStatus(str, Enum):
    """Document processing status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    ERROR = "error"


class StageStatus(str, Enum):
    """Individual stage status enumeration"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentLoadedEvent(Event):
    """
    Event triggered when a document is successfully loaded
    
    This event carries the basic document information and file path
    for further processing stages.
    """
    file_path: str = Field(..., description="Path to the loaded PDF file")
    filename: str = Field(..., description="Original filename of the document")
    file_size_mb: float = Field(..., description="File size in megabytes")
    document_id: str = Field(..., description="Unique identifier for this document")
    timestamp: float = Field(..., description="Timestamp when document was loaded")
    
    @validator('file_size_mb')
    def validate_file_size(cls, v):
        if v <= 0:
            raise ValueError('File size must be positive')
        if v > 50:  # Based on existing system limit
            raise ValueError('File size exceeds maximum limit of 50MB')
        return v


class DocumentParsedEvent(Event):
    """
    Event triggered when LlamaParse successfully processes a PDF
    
    This event carries the parsed Document objects and processing statistics.
    """
    documents: List[Document] = Field(..., description="List of parsed Document objects")
    file_path: str = Field(..., description="Original file path")
    document_id: str = Field(..., description="Unique identifier for this document")
    processing_time: float = Field(..., description="Time taken for parsing in seconds")
    parsing_stats: Dict[str, Any] = Field(default_factory=dict, description="Statistics from LlamaParse")
    timestamp: float = Field(..., description="Timestamp when parsing completed")
    
    @validator('documents')
    def validate_documents(cls, v):
        if not v:
            raise ValueError('At least one document must be parsed')
        return v
    
    @validator('processing_time')
    def validate_processing_time(cls, v):
        if v < 0:
            raise ValueError('Processing time cannot be negative')
        return v


class MetadataExtractedEvent(Event):
    """
    Event triggered when LlamaExtract successfully extracts metadata
    
    This event carries the extracted metadata along with the parsed documents.
    """
    documents: List[Document] = Field(..., description="List of Document objects from previous stage")
    metadata: Dict[str, Any] = Field(..., description="Extracted metadata dictionary")
    document_id: str = Field(..., description="Unique identifier for this document")
    file_path: str = Field(..., description="Original file path")
    extraction_time: float = Field(..., description="Time taken for metadata extraction in seconds")
    extraction_method: str = Field(..., description="Method used for extraction (LlamaExtract or fallback)")
    timestamp: float = Field(..., description="Timestamp when extraction completed")
    
    @validator('metadata')
    def validate_metadata(cls, v):
        required_fields = ['title', 'authors', 'abstract', 'mainFindings']
        for field in required_fields:
            if field not in v:
                raise ValueError(f'Required metadata field {field} is missing')
        return v


class DocumentChunkedEvent(Event):
    """
    Event triggered when document is successfully chunked
    
    This event carries the chunked nodes ready for vector indexing.
    """
    chunks: List[TextNode] = Field(..., description="List of chunked TextNode objects")
    original_documents: List[Document] = Field(..., description="Original documents before chunking")
    metadata: Dict[str, Any] = Field(..., description="Document metadata from previous stage")
    document_id: str = Field(..., description="Unique identifier for this document")
    file_path: str = Field(..., description="Original file path")
    chunking_time: float = Field(..., description="Time taken for chunking in seconds")
    chunking_method: str = Field(..., description="Chunking method used (hybrid or simplified)")
    chunk_stats: Dict[str, int] = Field(default_factory=dict, description="Chunking statistics")
    timestamp: float = Field(..., description="Timestamp when chunking completed")
    
    @validator('chunks')
    def validate_chunks(cls, v):
        if not v:
            raise ValueError('At least one chunk must be created')
        return v
    
    @validator('chunk_stats')
    def set_default_stats(cls, v, values):
        if 'chunks' in values:
            v.setdefault('total_chunks', len(values['chunks']))
            v.setdefault('avg_chunk_size', sum(len(chunk.text) for chunk in values['chunks']) // len(values['chunks']) if values['chunks'] else 0)
        return v


class DocumentIndexedEvent(Event):
    """
    Event triggered when document chunks are successfully indexed in vector store
    
    This event represents the completion of the document processing pipeline.
    """
    document_id: str = Field(..., description="Unique identifier for this document")
    file_path: str = Field(..., description="Original file path")
    filename: str = Field(..., description="Original filename")
    chunks_indexed: int = Field(..., description="Number of chunks successfully indexed")
    total_chunks: int = Field(..., description="Total number of chunks created")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")
    indexing_time: float = Field(..., description="Time taken for indexing in seconds")
    vector_ids: List[str] = Field(default_factory=list, description="List of vector IDs in Pinecone")
    index_stats: Dict[str, Any] = Field(default_factory=dict, description="Indexing statistics")
    timestamp: float = Field(..., description="Timestamp when indexing completed")
    
    @validator('chunks_indexed')
    def validate_indexed_count(cls, v, values):
        if v <= 0:
            raise ValueError('At least one chunk must be indexed')
        if 'total_chunks' in values and v > values['total_chunks']:
            raise ValueError('Cannot index more chunks than were created')
        return v


class DocumentProcessingErrorEvent(Event):
    """
    Event triggered when an error occurs during document processing
    
    This event carries error information and allows for graceful error handling.
    """
    document_id: str = Field(..., description="Unique identifier for the failed document")
    file_path: str = Field(..., description="Original file path")
    filename: str = Field(..., description="Original filename")
    error_stage: str = Field(..., description="Stage where the error occurred")
    error_message: str = Field(..., description="Detailed error message")
    error_type: str = Field(..., description="Type of error that occurred")
    partial_results: Optional[Dict[str, Any]] = Field(None, description="Any partial results before failure")
    timestamp: float = Field(..., description="Timestamp when error occurred")
    retry_count: int = Field(default=0, description="Number of retry attempts made")
    
    @validator('error_stage')
    def validate_error_stage(cls, v):
        valid_stages = ['loading', 'parsing', 'metadata_extraction', 'chunking', 'indexing', 'workflow', 'worker']
        if v not in valid_stages:
            raise ValueError(f'Error stage must be one of {valid_stages}')
        return v


class BatchProcessingStartEvent(Event):
    """
    Event triggered to start batch processing of multiple documents
    
    This event initializes the parallel processing system.
    """
    file_paths: List[str] = Field(..., description="List of file paths to process")
    batch_id: str = Field(..., description="Unique identifier for this batch")
    max_parallel_workers: int = Field(default=2, description="Maximum number of parallel workflows")
    processing_config: Dict[str, Any] = Field(default_factory=dict, description="Configuration for processing")
    timestamp: float = Field(..., description="Timestamp when batch processing started")
    
    @validator('file_paths')
    def validate_file_paths(cls, v):
        if not v:
            raise ValueError('At least one file path must be provided')
        if len(v) > 20:  # Based on system design limit
            raise ValueError('Cannot process more than 20 documents in a single batch')
        return v
    
    @validator('max_parallel_workers')
    def validate_workers(cls, v):
        if v < 1:
            raise ValueError('Must have at least 1 parallel worker')
        if v > 5:  # Reasonable upper limit to prevent resource exhaustion
            raise ValueError('Cannot have more than 5 parallel workers')
        return v


class BatchProcessingCompleteEvent(Event):
    """
    Event triggered when batch processing is complete
    
    This event carries the results of the entire batch processing operation.
    """
    batch_id: str = Field(..., description="Unique identifier for the completed batch")
    total_documents: int = Field(..., description="Total number of documents in the batch")
    successful_documents: int = Field(..., description="Number of successfully processed documents")
    failed_documents: int = Field(..., description="Number of failed documents")
    processing_time: float = Field(..., description="Total batch processing time in seconds")
    results: List[Union[DocumentIndexedEvent, DocumentProcessingErrorEvent]] = Field(..., description="Results for each document")
    batch_stats: Dict[str, Any] = Field(default_factory=dict, description="Batch processing statistics")
    timestamp: float = Field(..., description="Timestamp when batch processing completed")
    
    @validator('results')
    def validate_results_count(cls, v, values):
        if 'total_documents' in values and len(v) != values['total_documents']:
            raise ValueError('Number of results must match total documents')
        return v


class BatchProgressEvent(Event):
    """
    Event for reporting batch processing progress
    
    This event is used for real-time progress updates during batch processing.
    """
    batch_id: str = Field(..., description="Unique identifier for the batch")
    completed_documents: int = Field(..., description="Number of completed documents")
    total_documents: int = Field(..., description="Total number of documents in batch")
    active_workflows: int = Field(..., description="Number of currently active workflows")
    current_stage_info: Dict[str, Any] = Field(default_factory=dict, description="Information about current stages")
    estimated_completion_time: Optional[float] = Field(None, description="Estimated completion time in seconds")
    timestamp: float = Field(..., description="Timestamp of this progress update")
    
    @property
    def progress_percentage(self) -> float:
        """Calculate completion percentage"""
        if self.total_documents == 0:
            return 0.0
        return (self.completed_documents / self.total_documents) * 100.0


class DocumentStageProgress(BaseModel):
    """
    Model for tracking individual document progress through stages
    
    This is used for detailed monitoring and reporting.
    """
    document_id: str = Field(..., description="Unique identifier for the document")
    filename: str = Field(..., description="Original filename")
    file_path: str = Field(..., description="Original file path")
    current_stage: str = Field(..., description="Current processing stage")
    stages_status: Dict[str, StageStatus] = Field(default_factory=dict, description="Status of each stage")
    error_info: Optional[Dict[str, str]] = Field(None, description="Error information if any")
    processing_start_time: float = Field(..., description="When processing started for this document")
    stage_timings: Dict[str, float] = Field(default_factory=dict, description="Time taken for each completed stage")
    
    def update_stage_status(self, stage: str, status: StageStatus, timing: Optional[float] = None):
        """Update the status of a specific stage"""
        self.stages_status[stage] = status
        self.current_stage = stage
        if timing is not None:
            self.stage_timings[stage] = timing
        
        logger.debug(f"Updated {self.document_id} stage {stage} to {status}")
    
    def mark_error(self, stage: str, error_message: str, error_type: str):
        """Mark document as failed at a specific stage"""
        self.stages_status[stage] = StageStatus.FAILED
        self.current_stage = stage
        self.error_info = {
            'stage': stage,
            'message': error_message,
            'type': error_type
        }
        
        logger.error(f"Document {self.document_id} failed at {stage}: {error_message}")
    
    @property
    def is_completed(self) -> bool:
        """Check if document processing is completed successfully"""
        required_stages = ['loading', 'parsing', 'metadata_extraction', 'chunking', 'indexing']
        return all(self.stages_status.get(stage) == StageStatus.COMPLETED for stage in required_stages)
    
    @property
    def is_failed(self) -> bool:
        """Check if document processing has failed"""
        return StageStatus.FAILED in self.stages_status.values()
    
    @property
    def total_processing_time(self) -> float:
        """Calculate total processing time"""
        return sum(self.stage_timings.values())


# Event type unions for type hints
ProcessingEvent = Union[
    DocumentLoadedEvent,
    DocumentParsedEvent,
    MetadataExtractedEvent,
    DocumentChunkedEvent,
    DocumentIndexedEvent,
    DocumentProcessingErrorEvent
]

BatchEvent = Union[
    BatchProcessingStartEvent,
    BatchProcessingCompleteEvent,
    BatchProgressEvent
]

WorkflowEvent = Union[ProcessingEvent, BatchEvent]


# Helper functions for event creation
def create_document_loaded_event(
    file_path: str,
    filename: str,
    file_size_mb: float,
    document_id: str,
    timestamp: float
) -> DocumentLoadedEvent:
    """Helper function to create a DocumentLoadedEvent"""
    return DocumentLoadedEvent(
        file_path=file_path,
        filename=filename,
        file_size_mb=file_size_mb,
        document_id=document_id,
        timestamp=timestamp
    )


def create_error_event(
    document_id: str,
    file_path: str,
    filename: str,
    error_stage: str,
    error_message: str,
    error_type: str,
    timestamp: float,
    partial_results: Optional[Dict[str, Any]] = None,
    retry_count: int = 0
) -> DocumentProcessingErrorEvent:
    """Helper function to create a DocumentProcessingErrorEvent"""
    return DocumentProcessingErrorEvent(
        document_id=document_id,
        file_path=file_path,
        filename=filename,
        error_stage=error_stage,
        error_message=error_message,
        error_type=error_type,
        partial_results=partial_results,
        timestamp=timestamp,
        retry_count=retry_count
    )


# Initialize default stage tracking
DEFAULT_STAGES = ['loading', 'parsing', 'metadata_extraction', 'chunking', 'indexing']


def initialize_document_progress(
    document_id: str,
    filename: str,
    file_path: str,
    start_time: float
) -> DocumentStageProgress:
    """Initialize progress tracking for a new document"""
    stages_status = {stage: StageStatus.NOT_STARTED for stage in DEFAULT_STAGES}
    
    return DocumentStageProgress(
        document_id=document_id,
        filename=filename,
        file_path=file_path,
        current_stage='loading',
        stages_status=stages_status,
        processing_start_time=start_time,
        stage_timings={}
    )


logger.info("Workflow events module initialized successfully")