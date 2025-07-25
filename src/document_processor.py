"""
Document processor module using LlamaParse for PDF processing

This module provides PDF processing capabilities using LlamaParse to extract
text content, formulas in LaTeX format, and table structure preservation.
Based on PRD specifications and LlamaIndex integration patterns.
"""

import os
import asyncio
from typing import List, Optional, Union
import nest_asyncio
from llama_parse import LlamaParse
from llama_index.core import Document
from pydantic import BaseModel, Field

from .config import settings

# Apply nest_asyncio for notebook compatibility
nest_asyncio.apply()



def _run_async_safely(coro):
    """
    Safely run async coroutine, handling closed event loops in parallel processing
    
    Args:
        coro: Coroutine to execute
        
    Returns:
        Result of coroutine execution
    """
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
        
        # Check if we're already in an async context
        if loop.is_running():
            # Use nest_asyncio to run in already running loop
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        else:
            # Run in the existing but not running loop
            return loop.run_until_complete(coro)
    except (RuntimeError, AttributeError) as e:
        # Create a new event loop if the current one is closed or doesn't exist
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            result = new_loop.run_until_complete(coro)
            return result
        finally:
            new_loop.close()
            # Try to restore original loop if possible
            try:
                asyncio.set_event_loop(loop)
            except:
                pass


class DocumentProcessorConfig(BaseModel):
    """Configuration for DocumentProcessor"""
    
    api_key: str = Field(default_factory=lambda: settings.llama_cloud_api_key)
    result_type: str = Field(default=settings.llamaparse_result_type)  # "markdown" or "text"
    language: str = Field(default=settings.llamaparse_language)
    # New prompt system (replaces deprecated parsing_instruction)
    system_prompt_append: str = Field(default=settings.llamaparse_system_prompt_append)
    user_prompt: Optional[str] = Field(default=settings.llamaparse_user_prompt)
    verbose: bool = Field(default=True)
    fast_mode: bool = Field(default=False)


class DocumentProcessor:
    """
    PDF document processor using LlamaParse
    
    This class handles PDF processing with advanced extraction capabilities:
    - Text content extraction with structure preservation
    - Formula conversion to LaTeX format
    - Table extraction with structure maintained
    - Image and diagram processing
    
    Based on PRD specifications for ArXiv scientific document analysis.
    """
    
    def __init__(self, config: Optional[DocumentProcessorConfig] = None):
        """
        Initialize DocumentProcessor with LlamaParse integration
        
        Args:
            config: Optional configuration override
        """
        self.config = config or DocumentProcessorConfig()
        
        # Validate API key
        if not self.config.api_key or self.config.api_key.startswith("your_"):
            raise ValueError(
                "LLAMA_CLOUD_API_KEY is required. "
                "Set it in environment variables or .env file"
            )
        
        # Initialize LlamaParse with configuration (using new prompt system)
        parser_kwargs = {
            "api_key": self.config.api_key,
            "result_type": self.config.result_type,
            "language": self.config.language,
            "system_prompt_append": self.config.system_prompt_append,
            "verbose": self.config.verbose,
            "fast_mode": self.config.fast_mode
        }
        
        # Add user_prompt only if it's not None
        if self.config.user_prompt:
            parser_kwargs["user_prompt"] = self.config.user_prompt
        
        self.parser = LlamaParse(**parser_kwargs)
        
    
    def process_pdf(self, file_path: Union[str, List[str]]) -> List[Document]:
        """
        Process PDF file(s) using LlamaParse with safe async handling
        
        Args:
            file_path: Path to PDF file or list of paths for batch processing
            
        Returns:
            List of Document objects with extracted content
            
        Raises:
            Exception: If PDF processing fails
        """
        try:
            
            # Use async version with safe event loop handling
            documents = _run_async_safely(self.process_pdf_async(file_path))
            
            return documents
            
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
    
    async def process_pdf_async(self, file_path: Union[str, List[str]]) -> List[Document]:
        """
        Asynchronously process PDF file(s) using LlamaParse
        
        Args:
            file_path: Path to PDF file or list of paths for batch processing
            
        Returns:
            List of Document objects with extracted content
            
        Raises:
            Exception: If PDF processing fails
        """
        try:
            
            # Process single file or batch asynchronously
            if isinstance(file_path, str):
                documents = await self.parser.aload_data(file_path)
            else:
                documents = await self.parser.aload_data(file_path)
            
            
            # Add processing metadata
            for i, doc in enumerate(documents):
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    doc.metadata = {}
                
                doc.metadata.update({
                    "source_file": file_path if isinstance(file_path, str) else file_path[i] if i < len(file_path) else "batch_processing",
                    "processing_method": "LlamaParse_async",
                    "result_type": self.config.result_type,
                    "document_index": i
                })
            
            return documents
            
        except Exception as e:
            raise Exception(f"Error processing PDF (async): {str(e)}")
    
    def get_json_result(self, file_path: str) -> List[dict]:
        """
        Get detailed JSON result with page-level information
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of dictionaries containing detailed page information
            
        Raises:
            Exception: If JSON extraction fails
        """
        try:
            
            json_results = self.parser.get_json_result(file_path)
            
            
            return json_results
            
        except Exception as e:
            raise Exception(f"Error getting JSON result: {str(e)}")
    
    def get_images(self, json_results: List[dict], download_path: str = "extracted_images") -> List[dict]:
        """
        Extract and download images from parsed document
        
        Args:
            json_results: JSON results from get_json_result()
            download_path: Directory to save extracted images
            
        Returns:
            List of image dictionaries with metadata
            
        Raises:
            Exception: If image extraction fails
        """
        try:
            
            # Create download directory if it doesn't exist
            os.makedirs(download_path, exist_ok=True)
            
            image_dicts = self.parser.get_images(json_results, download_path=download_path)
            
            
            return image_dicts
            
        except Exception as e:
            raise Exception(f"Error extracting images: {str(e)}")
    
    def create_documents_from_json(self, json_results: List[dict]) -> List[Document]:
        """
        Create Document objects from JSON results with page-level granularity
        
        Args:
            json_results: JSON results from get_json_result()
            
        Returns:
            List of Document objects, one per page
        """
        documents = []
        
        for doc_idx, document_json in enumerate(json_results):
            for page in document_json.get("pages", []):
                page_text = page.get("text", "")
                page_number = page.get("page", 0)
                
                # Create Document with page-level metadata
                doc = Document(
                    text=page_text,
                    metadata={
                        "page_number": page_number,
                        "document_index": doc_idx,
                        "processing_method": "LlamaParse_JSON",
                        "result_type": "page_level",
                        # Include any additional metadata from the page
                        **{k: v for k, v in page.items() if k not in ["text", "page"]}
                    }
                )
                
                documents.append(doc)
        
        
        return documents
    
    def validate_processing_result(self, documents: List[Document]) -> bool:
        """
        Validate that processing results meet quality standards
        
        Args:
            documents: List of processed documents
            
        Returns:
            True if validation passes, False otherwise
        """
        if not documents:
            return False
        
        for i, doc in enumerate(documents):
            # Check for minimum content length
            if len(doc.text.strip()) < 10:
                return False
            
            # Check for metadata presence
            if not hasattr(doc, 'metadata') or not doc.metadata:
                return False
        
        return True
    
    def get_processing_stats(self, documents: List[Document]) -> dict:
        """
        Get statistics about processed documents
        
        Args:
            documents: List of processed documents
            
        Returns:
            Dictionary with processing statistics
        """
        if not documents:
            return {"total_documents": 0}
        
        total_chars = sum(len(doc.text) for doc in documents)
        total_words = sum(len(doc.text.split()) for doc in documents)
        
        stats = {
            "total_documents": len(documents),
            "total_characters": total_chars,
            "total_words": total_words,
            "average_chars_per_doc": total_chars / len(documents),
            "average_words_per_doc": total_words / len(documents),
            "result_type": self.config.result_type,
            "processing_method": "LlamaParse"
        }
        
        
        return stats


class ArxivDocumentProcessor:
    """
    Unified document processor that combines PDF processing, metadata extraction, and chunking
    
    This is the main class that orchestrates the entire document processing pipeline
    for ArXiv scientific publications according to PRD specifications.
    """
    
    def __init__(self, config: Optional[DocumentProcessorConfig] = None):
        """
        Initialize the unified ArXiv document processor
        
        Args:
            config: Optional configuration override
        """
        self.doc_processor = DocumentProcessor(config)
        # Note: metadata_extractor and chunker will be initialized when those modules are ready
    
    def process(self, file_path: Union[str, List[str]]) -> dict:
        """
        Process PDF file(s) end-to-end
        
        Args:
            file_path: Path to PDF file or list of paths
            
        Returns:
            Dictionary containing documents, metadata, and processing stats
        """
        try:
            
            # Step 1: Extract text with LlamaParse
            documents = self.doc_processor.process_pdf(file_path)
            
            # Validate processing
            if not self.doc_processor.validate_processing_result(documents):
                raise Exception("Document processing validation failed")
            
            # Get processing statistics
            stats = self.doc_processor.get_processing_stats(documents)
            
            result = {
                "documents": documents,
                "processing_stats": stats,
                "file_path": file_path,
                "success": True
            }
            
            
            return result
            
        except Exception as e:
            return {
                "documents": [],
                "processing_stats": {},
                "file_path": file_path,
                "success": False,
                "error": str(e)
            }


# Convenience functions for easy usage
def process_pdf(file_path: Union[str, List[str]], **kwargs) -> List[Document]:
    """
    Convenience function to process PDF with default settings
    
    Args:
        file_path: Path to PDF file or list of paths
        **kwargs: Additional configuration options
        
    Returns:
        List of processed Document objects
    """
    config = DocumentProcessorConfig(**kwargs)
    processor = DocumentProcessor(config)
    return processor.process_pdf(file_path)


async def process_pdf_async(file_path: Union[str, List[str]], **kwargs) -> List[Document]:
    """
    Convenience function to process PDF asynchronously with default settings
    
    Args:
        file_path: Path to PDF file or list of paths
        **kwargs: Additional configuration options
        
    Returns:
        List of processed Document objects
    """
    config = DocumentProcessorConfig(**kwargs)
    processor = DocumentProcessor(config)
    return await processor.process_pdf_async(file_path)