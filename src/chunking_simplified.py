"""
Simplified chunking module for uniform content processing

This module provides a simplified chunking strategy that processes all content uniformly:
- All content: fixed size with overlap using SentenceSplitter
- Preserves all existing metadata structure
- Maintains compatibility with existing interfaces
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from .config import settings
from .logging_config import log_chunks_created



class ChunkingConfig(BaseModel):
    """Configuration for simplified chunking"""
    
    chunk_size: int = Field(default=settings.max_chunk_size)  # 1024 tokens
    chunk_overlap: float = Field(default=settings.chunk_overlap)  # 0.15 (15%)
    min_chunk_size: int = Field(default=50)  # Minimum chunk size
    preserve_sentences: bool = Field(default=True)  # Preserve sentence boundaries


class ChunkMetadata(BaseModel):
    """Metadata for a chunk"""
    
    page_number: int = Field(default=0)
    original_text_length: int = Field(default=0)
    
    # Context information
    previous_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    parent_section: Optional[str] = None


class HybridChunker:
    """
    Simplified chunking strategy for uniform content processing
    
    This class implements a simplified approach that processes all content
    uniformly using SentenceSplitter while maintaining full metadata compatibility.
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize simplified chunker
        
        Args:
            config: Optional configuration override
        """
        self.config = config or ChunkingConfig()
        
        # Initialize sentence splitter for all content
        self.sentence_splitter = SentenceSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=int(self.config.chunk_size * self.config.chunk_overlap),
            separator=" "
        )
        
    
    def chunk_document(self, document: Document, metadata: Optional[Dict[str, Any]] = None) -> List[TextNode]:
        """
        Apply simplified chunking strategy to document
        
        Args:
            document: Document to chunk
            metadata: Optional document-level metadata
            
        Returns:
            List of TextNode objects with appropriate metadata
        """
        try:
            
            # Initialize metadata
            doc_metadata = metadata or {}
            if hasattr(document, 'metadata') and document.metadata:
                doc_metadata.update(document.metadata)
            
            # Direct text splitting using SentenceSplitter
            text_chunks = self.sentence_splitter.split_text(document.text)
            
            chunks = []
            for i, chunk_text in enumerate(text_chunks):
                # Create TextNode with full metadata
                chunk_node = self._create_text_node(
                    text=chunk_text,
                    chunk_type="text",
                    section_title="Document",
                    page_number=self._estimate_page_number(chunk_text, document.text),
                    chunk_index=i,
                    doc_metadata=doc_metadata
                )
                chunks.append(chunk_node)
            
            # Preserve all metadata functions
            self._add_chunk_relationships(chunks)
            self.log_chunk_structure(chunks)
            stats = self.get_chunking_stats(chunks)
            
            # Log chunk IDs for tracking
            chunk_ids = [chunk.id_ for chunk in chunks]
            filename = getattr(document, 'metadata', {}).get('source_file', 'unknown')
            if filename == 'unknown' and doc_metadata:
                filename = doc_metadata.get('filename', 'unknown')
            log_chunks_created(filename, chunk_ids)
            
            return chunks
            
        except Exception as e:
            raise
    
    def _estimate_page_number(self, chunk_text: str, full_text: str) -> int:
        """
        Estimate page number based on chunk position in document
        
        Args:
            chunk_text: Text of the chunk
            full_text: Full document text
            
        Returns:
            Estimated page number
        """
        chunk_position = full_text.find(chunk_text)
        if chunk_position == -1:
            return 1
        
        # Approximately 2000 characters per page
        chars_per_page = 2000
        return max(1, (chunk_position // chars_per_page) + 1)
    
    def _create_text_node(self, text: str, chunk_type: str, section_title: str,
                         page_number: int, chunk_index: int, doc_metadata: Dict[str, Any],
                         additional_metadata: Optional[Dict[str, Any]] = None) -> TextNode:
        """
        Create TextNode with proper metadata
        
        Args:
            text: Chunk text
            chunk_type: Type of chunk
            section_title: Section title
            page_number: Page number
            chunk_index: Chunk index
            doc_metadata: Document metadata
            additional_metadata: Additional chunk-specific metadata
            
        Returns:
            TextNode with metadata
        """
        # Create chunk metadata
        chunk_metadata = ChunkMetadata(
            page_number=page_number,
            original_text_length=len(text)
        )
        
        # Combine all metadata
        full_metadata = {
            **doc_metadata,  # Document-level metadata
            **chunk_metadata.model_dump(),  # Chunk-level metadata
            **(additional_metadata or {})  # Additional metadata
        }
        
        # Generate chunk ID based on text hash and index
        text_hash = abs(hash(text[:50])) % 1000000  # Keep it shorter and positive
        chunk_id = f"chunk_{text_hash}_{chunk_index}"
        
        # Create TextNode
        node = TextNode(
            text=text,
            metadata=full_metadata,
            id_=chunk_id,
        )
        
        # Log chunk creation
        
        return node
    
    def _add_chunk_relationships(self, chunks: List[TextNode]):
        """
        Add relationships between chunks (previous/next)
        
        Args:
            chunks: List of chunks to process
        """
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.metadata["previous_chunk_id"] = chunks[i-1].id_
            if i < len(chunks) - 1:
                chunk.metadata["next_chunk_id"] = chunks[i+1].id_
    
    def get_chunking_stats(self, chunks: List[TextNode]) -> Dict[str, Any]:
        """
        Get statistics about chunking results
        
        Args:
            chunks: List of chunks
            
        Returns:
            Dictionary with chunking statistics
        """
        if not chunks:
            return {"total_chunks": 0}
        
        total_chars = sum(len(chunk.text) for chunk in chunks)
        
        stats = {
            "total_chunks": len(chunks),
            "total_characters": total_chars,
            "average_chars_per_chunk": total_chars / len(chunks),
            "chunk_size_config": self.config.chunk_size,
            "chunk_overlap_config": self.config.chunk_overlap
        }
        
        return stats
    
    def log_chunk_structure(self, chunks: List[TextNode]):
        """
        Log detailed structure of created chunks for debugging and monitoring
        
        Args:
            chunks: List of created chunks
        """
        if not chunks:
            return
        
        
        for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
            pass  # Basic logging handled by log_chunks_created
        
        if len(chunks) > 5:
            pass  # Additional chunks logged in log_chunks_created
        


# Convenience function for backward compatibility
def chunk_document(document: Document, **kwargs) -> List[TextNode]:
    """
    Convenience function to chunk document with default settings
    
    Args:
        document: Document to chunk
        **kwargs: Additional arguments passed to HybridChunker
        
    Returns:
        List of TextNode objects
    """
    chunker = HybridChunker()
    return chunker.chunk_document(document, **kwargs)