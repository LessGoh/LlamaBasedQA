"""
Document Processing Workflow Module for ArXiv RAG System

This module implements the main DocumentProcessingWorkflow that orchestrates
the entire document processing pipeline using LlamaIndex Workflow system.

The workflow wraps existing processing modules in workflow steps to enable
parallel processing while maintaining all existing business logic.

Based on the plan for creating a parallel document processing system with queue.
"""

import logging
import os
import tempfile
import time
import uuid
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

from llama_index.core.workflow import (
    Workflow,
    StartEvent,
    StopEvent,
    Context,
    step
)
from llama_index.core import Document

# Import our custom events
from .workflow_events import (
    DocumentLoadedEvent,
    DocumentParsedEvent,
    MetadataExtractedEvent,
    DocumentChunkedEvent,
    DocumentIndexedEvent,
    DocumentProcessingErrorEvent,
    create_error_event,
    StageStatus
)

# Import existing modules (will be integrated in phase 1.3)
from .document_processor import ArxivDocumentProcessor
from .metadata_extractor import MetadataExtractor
from .chunking_simplified import HybridChunker
from .vector_store import DocumentIndexer
from .utils import FileValidator, metrics, format_processing_time

# Set up logger
logger = logging.getLogger(__name__)


class DocumentProcessingWorkflow(Workflow):
    """
    Main workflow for processing individual documents through the complete pipeline
    
    This workflow orchestrates:
    1. Document loading and validation
    2. PDF parsing with LlamaParse
    3. Metadata extraction with LlamaExtract
    4. Document chunking with HybridChunker
    5. Vector indexing with Pinecone
    
    Each step is wrapped in error handling for robust parallel processing.
    """
    
    def __init__(self, timeout: float = 1800.0, verbose: bool = False):
        """
        Initialize the document processing workflow
        
        Args:
            timeout: Maximum time in seconds for workflow completion (default 30 minutes)
            verbose: Whether to enable verbose logging
        """
        super().__init__(timeout=timeout, verbose=verbose)
        
        # Initialize processors (will be done properly in integration phase)
        self.doc_processor = None
        self.metadata_processor = None
        self.chunker = None
        self.indexer = None
        
        logger.info(f"DocumentProcessingWorkflow initialized with timeout={timeout}s, verbose={verbose}")
    
    @step
    async def load_document(self, ctx: Context, ev: StartEvent) -> DocumentLoadedEvent | StopEvent:
        """
        Load and validate a PDF document
        
        Args:
            ctx: Workflow context for state management
            ev: StartEvent containing file_path and other parameters
            
        Returns:
            DocumentLoadedEvent on success, DocumentProcessingErrorEvent on failure
        """
        try:
            file_path = ev.get("file_path")
            if not file_path:
                raise ValueError("file_path is required in StartEvent")
            
            # Generate unique document ID
            document_id = str(uuid.uuid4())
            filename = Path(file_path).name
            start_time = time.time()
            
            # Store document info in context for later steps
            await ctx.store.set("document_id", document_id)
            await ctx.store.set("file_path", file_path)
            await ctx.store.set("filename", filename)
            await ctx.store.set("start_time", start_time)
            
            # Validate file exists and is readable
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Read and validate PDF
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            if not FileValidator.validate_pdf(file_content):
                raise ValueError("Invalid PDF file format")
            
            file_size_mb = FileValidator.get_file_size_mb(file_content)
            if not FileValidator.validate_file_size(file_content, max_size_mb=50):
                raise ValueError(f"File too large ({file_size_mb:.1f}MB). Maximum size is 50MB")
            
            # Store file content for next step if needed
            await ctx.store.set("file_content", file_content)
            
            logger.info(f"Document loaded successfully: {filename} ({file_size_mb:.1f}MB)")
            
            return DocumentLoadedEvent(
                file_path=file_path,
                filename=filename,
                file_size_mb=file_size_mb,
                document_id=document_id,
                timestamp=time.time()
            )
            
        except Exception as e:
            error_msg = f"Failed to load document: {str(e)}"
            logger.error(f"Error in load_document: {error_msg}")
            
            error_event = create_error_event(
                document_id=await ctx.store.get("document_id", str(uuid.uuid4())),
                file_path=ev.get("file_path", "unknown"),
                filename=Path(ev.get("file_path", "unknown")).name,
                error_stage="loading",
                error_message=error_msg,
                error_type=type(e).__name__,
                timestamp=time.time()
            )
            return StopEvent(result=error_event)
    
    @step
    async def parse_document(self, ctx: Context, ev: DocumentLoadedEvent) -> DocumentParsedEvent | StopEvent:
        """
        Parse PDF document using LlamaParse
        
        Args:
            ctx: Workflow context
            ev: DocumentLoadedEvent from previous step
            
        Returns:
            DocumentParsedEvent on success, DocumentProcessingErrorEvent on failure
        """
        try:
            parse_start_time = time.time()
            
            # Initialize document processor if not already done
            if self.doc_processor is None:
                self.doc_processor = ArxivDocumentProcessor()
            
            logger.info(f"Starting PDF parsing for: {ev.filename}")
            
            # Process PDF with LlamaParse
            processing_result = self.doc_processor.process(ev.file_path)
            
            if not processing_result.get("success", False):
                raise Exception(f"Document processing failed: {processing_result.get('error', 'Unknown error')}")
            
            documents = processing_result.get("documents", [])
            if not documents:
                raise Exception("No documents were extracted from PDF")
            
            processing_time = time.time() - parse_start_time
            
            # Store documents in context for next steps
            await ctx.store.set("documents", documents)
            
            # Record metrics
            metrics.record_timing("pdf_parsing", processing_time)
            metrics.increment_counter("pdfs_parsed")
            
            logger.info(f"PDF parsing completed: {ev.filename} in {format_processing_time(processing_time)}")
            
            return DocumentParsedEvent(
                documents=documents,
                file_path=ev.file_path,
                document_id=ev.document_id,
                processing_time=processing_time,
                parsing_stats=processing_result.get("processing_stats", {}),
                timestamp=time.time()
            )
            
        except Exception as e:
            error_msg = f"Failed to parse PDF: {str(e)}"
            logger.error(f"Error in parse_document for {ev.filename}: {error_msg}")
            
            error_event = create_error_event(
                document_id=ev.document_id,
                file_path=ev.file_path,
                filename=ev.filename,
                error_stage="parsing",
                error_message=error_msg,
                error_type=type(e).__name__,
                timestamp=time.time(),
                partial_results={"file_size_mb": ev.file_size_mb}
            )
            return StopEvent(result=error_event)
    
    @step
    async def extract_metadata(self, ctx: Context, ev: DocumentParsedEvent) -> MetadataExtractedEvent | StopEvent:
        """
        Extract structured metadata using LlamaExtract
        
        Args:
            ctx: Workflow context
            ev: DocumentParsedEvent from previous step
            
        Returns:
            MetadataExtractedEvent on success, DocumentProcessingErrorEvent on failure
        """
        try:
            extraction_start_time = time.time()
            
            # Initialize metadata processor if not already done
            if self.metadata_processor is None:
                self.metadata_processor = MetadataExtractor()
            
            logger.info(f"Starting metadata extraction for: {Path(ev.file_path).name}")
            
            # Combine all documents into one for metadata extraction
            full_text = "\n\n".join([doc.text for doc in ev.documents])
            full_document = Document(text=full_text, metadata=ev.documents[0].metadata)
            
            # Extract metadata
            extracted_metadata = self.metadata_processor.extract_metadata(full_document)
            
            # Convert to dictionary for compatibility
            if hasattr(extracted_metadata, 'model_dump'):
                metadata_dict = extracted_metadata.model_dump()
            elif hasattr(extracted_metadata, 'dict'):
                metadata_dict = extracted_metadata.dict()
            else:
                metadata_dict = extracted_metadata
            
            extraction_time = time.time() - extraction_start_time
            
            # Store metadata in context for next steps
            await ctx.store.set("metadata", metadata_dict)
            
            # Record metrics
            metrics.record_timing("metadata_extraction", extraction_time)
            metrics.increment_counter("metadata_extracted")
            
            extraction_method = "LlamaExtract" if not self.metadata_processor.use_fallback else "fallback"
            
            logger.info(f"Metadata extraction completed for {Path(ev.file_path).name} using {extraction_method} in {format_processing_time(extraction_time)}")
            
            return MetadataExtractedEvent(
                documents=ev.documents,
                metadata=metadata_dict,
                document_id=ev.document_id,
                file_path=ev.file_path,
                extraction_time=extraction_time,
                extraction_method=extraction_method,
                timestamp=time.time()
            )
            
        except Exception as e:
            error_msg = f"Failed to extract metadata: {str(e)}"
            logger.error(f"Error in extract_metadata for {Path(ev.file_path).name}: {error_msg}")
            
            error_event = create_error_event(
                document_id=ev.document_id,
                file_path=ev.file_path,
                filename=Path(ev.file_path).name,
                error_stage="metadata_extraction",
                error_message=error_msg,
                error_type=type(e).__name__,
                timestamp=time.time(),
                partial_results={
                    "parsing_time": ev.processing_time,
                    "documents_count": len(ev.documents)
                }
            )
            return StopEvent(result=error_event)
    
    @step
    async def chunk_document(self, ctx: Context, ev: MetadataExtractedEvent) -> DocumentChunkedEvent | StopEvent:
        """
        Chunk document using HybridChunker
        
        Args:
            ctx: Workflow context
            ev: MetadataExtractedEvent from previous step
            
        Returns:
            DocumentChunkedEvent on success, DocumentProcessingErrorEvent on failure
        """
        try:
            chunking_start_time = time.time()
            
            # Initialize chunker if not already done
            if self.chunker is None:
                self.chunker = HybridChunker()
            
            logger.info(f"Starting document chunking for: {Path(ev.file_path).name}")
            
            # Combine documents into one for chunking
            full_text = "\n\n".join([doc.text for doc in ev.documents])
            full_document = Document(text=full_text, metadata=ev.documents[0].metadata)
            
            # Chunk the document
            chunks = self.chunker.chunk_document(full_document, ev.metadata)
            
            if not chunks:
                raise Exception("No chunks were created from document")
            
            chunking_time = time.time() - chunking_start_time
            
            # Calculate chunk statistics
            chunk_stats = {
                "total_chunks": len(chunks),
                "avg_chunk_size": sum(len(chunk.text) for chunk in chunks) // len(chunks),
                "min_chunk_size": min(len(chunk.text) for chunk in chunks),
                "max_chunk_size": max(len(chunk.text) for chunk in chunks)
            }
            
            # Store chunks in context for next step
            await ctx.store.set("chunks", chunks)
            
            # Record metrics
            metrics.record_timing("document_chunking", chunking_time)
            metrics.increment_counter("documents_chunked")
            
            logger.info(f"Document chunking completed for {Path(ev.file_path).name}: {len(chunks)} chunks in {format_processing_time(chunking_time)}")
            
            return DocumentChunkedEvent(
                chunks=chunks,
                original_documents=ev.documents,
                metadata=ev.metadata,
                document_id=ev.document_id,
                file_path=ev.file_path,
                chunking_time=chunking_time,
                chunking_method="hybrid",
                chunk_stats=chunk_stats,
                timestamp=time.time()
            )
            
        except Exception as e:
            error_msg = f"Failed to chunk document: {str(e)}"
            logger.error(f"Error in chunk_document for {Path(ev.file_path).name}: {error_msg}")
            
            error_event = create_error_event(
                document_id=ev.document_id,
                file_path=ev.file_path,
                filename=Path(ev.file_path).name,
                error_stage="chunking",
                error_message=error_msg,
                error_type=type(e).__name__,
                timestamp=time.time(),
                partial_results={
                    "parsing_time": ev.extraction_time,
                    "metadata_fields": list(ev.metadata.keys()) if ev.metadata else []
                }
            )
            return StopEvent(result=error_event)
    
    @step
    async def index_chunks(self, ctx: Context, ev: DocumentChunkedEvent) -> StopEvent:
        """
        Index document chunks in vector store
        
        Args:
            ctx: Workflow context
            ev: DocumentChunkedEvent from previous step
            
        Returns:
            DocumentIndexedEvent on success (wrapped in StopEvent), DocumentProcessingErrorEvent on failure
        """
        try:
            indexing_start_time = time.time()
            
            # Initialize indexer if not already done
            if self.indexer is None:
                self.indexer = DocumentIndexer()
            
            logger.info(f"Starting vector indexing for: {Path(ev.file_path).name} ({len(ev.chunks)} chunks)")
            
            # Index chunks in vector store
            indexing_result = self.indexer.index_processed_chunks(ev.chunks)
            
            if not indexing_result.get("success", False):
                raise Exception(f"Indexing failed: {indexing_result.get('error', 'Unknown error')}")
            
            indexing_time = time.time() - indexing_start_time
            indexed_count = indexing_result.get("indexed_count", 0)
            
            if indexed_count == 0:
                raise Exception("No chunks were successfully indexed")
            
            # Calculate total processing time
            start_time = await ctx.store.get("start_time", time.time())
            total_time = time.time() - start_time
            
            # Record final metrics
            metrics.record_timing("vector_indexing", indexing_time)
            metrics.record_timing("total_document_processing", total_time)
            metrics.increment_counter("documents_indexed")
            
            logger.info(f"Document processing completed for {Path(ev.file_path).name}: {indexed_count}/{len(ev.chunks)} chunks indexed in {format_processing_time(total_time)} total")
            
            # Create the final success event
            success_event = DocumentIndexedEvent(
                document_id=ev.document_id,
                file_path=ev.file_path,
                filename=Path(ev.file_path).name,
                chunks_indexed=indexed_count,
                total_chunks=len(ev.chunks),
                metadata=ev.metadata,
                indexing_time=indexing_time,
                vector_ids=indexing_result.get("vector_ids", []),
                index_stats=indexing_result.get("stats", {}),
                timestamp=time.time()
            )
            
            return StopEvent(result=success_event)
            
        except Exception as e:
            error_msg = f"Failed to index chunks: {str(e)}"
            logger.error(f"Error in index_chunks for {Path(ev.file_path).name}: {error_msg}")
            
            error_event = create_error_event(
                document_id=ev.document_id,
                file_path=ev.file_path,
                filename=Path(ev.file_path).name,
                error_stage="indexing",
                error_message=error_msg,
                error_type=type(e).__name__,
                timestamp=time.time(),
                partial_results={
                    "chunks_created": len(ev.chunks),
                    "chunk_stats": ev.chunk_stats
                }
            )
            
            return StopEvent(result=error_event)
    
    def initialize_processors(self):
        """
        Initialize all processing components
        
        This method will be called during workflow setup to ensure
        all processors are ready for document processing.
        """
        try:
            if self.doc_processor is None:
                self.doc_processor = ArxivDocumentProcessor()
                logger.info("ArxivDocumentProcessor initialized")
            
            if self.metadata_processor is None:
                self.metadata_processor = MetadataExtractor()
                logger.info("MetadataExtractor initialized")
            
            if self.chunker is None:
                self.chunker = HybridChunker()
                logger.info("HybridChunker initialized")
            
            if self.indexer is None:
                self.indexer = DocumentIndexer()
                logger.info("DocumentIndexer initialized")
            
            logger.info("All workflow processors initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize processors: {str(e)}")
            raise
    
    async def process_document(self, file_path: str, **kwargs) -> Union[DocumentIndexedEvent, DocumentProcessingErrorEvent]:
        """
        Convenience method to process a single document through the entire workflow
        
        Args:
            file_path: Path to the PDF file to process
            **kwargs: Additional parameters for the workflow
            
        Returns:
            DocumentIndexedEvent on success, DocumentProcessingErrorEvent on failure
        """
        try:
            # Initialize processors
            self.initialize_processors()
            
            # Run the workflow
            result = await self.run(file_path=file_path, **kwargs)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return create_error_event(
                document_id=str(uuid.uuid4()),
                file_path=file_path,
                filename=Path(file_path).name,
                error_stage="workflow",
                error_message=f"Workflow error: {str(e)}",
                error_type=type(e).__name__,
                timestamp=time.time()
            )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the workflow processing
        
        Returns:
            Dictionary with processing statistics
        """
        return {
            "workflow_timeout": self.timeout,
            "processors_initialized": {
                "doc_processor": self.doc_processor is not None,
                "metadata_processor": self.metadata_processor is not None,
                "chunker": self.chunker is not None,
                "indexer": self.indexer is not None
            },
            "workflow_steps": [
                "load_document",
                "parse_document", 
                "extract_metadata",
                "chunk_document",
                "index_chunks"
            ]
        }


# Convenience function for easy workflow usage
async def process_single_document(file_path: str, timeout: float = 1800.0, verbose: bool = False) -> Union[DocumentIndexedEvent, DocumentProcessingErrorEvent]:
    """
    Process a single document through the complete workflow
    
    Args:
        file_path: Path to the PDF file to process
        timeout: Maximum processing time in seconds
        verbose: Enable verbose logging
        
    Returns:
        DocumentIndexedEvent on success, DocumentProcessingErrorEvent on failure
    """
    workflow = DocumentProcessingWorkflow(timeout=timeout, verbose=verbose)
    return await workflow.process_document(file_path)


logger.info("Document workflow module initialized successfully")