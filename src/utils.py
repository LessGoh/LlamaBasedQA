"""
Utility functions and helpers for the RAG system

This module provides utility functions used across the RAG system including
caching, formatting, validation, and helper classes.
"""

import logging
import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import pickle

# Set up logger
logger = logging.getLogger(__name__)


class QueryCache:
    """
    Simple in-memory cache for query results with TTL support
    """
    
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 100):
        """
        Initialize query cache
        
        Args:
            ttl_seconds: Time to live for cached items in seconds
            max_size: Maximum number of items to cache
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        logger.info(f"QueryCache initialized with TTL: {ttl_seconds}s, max_size: {max_size}")
    
    def _create_key(self, query: str, filters: Optional[str] = None) -> str:
        """Create cache key from query and filters"""
        key_data = f"{query}:{filters or ''}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _cleanup_expired(self):
        """Remove expired items from cache"""
        current_time = time.time()
        expired_keys = []
        
        for key, item in self.cache.items():
            if current_time - item["timestamp"] > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
            del self.access_times[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache items")
    
    def _evict_lru(self):
        """Evict least recently used items if cache is full"""
        while len(self.cache) >= self.max_size:
            # Find least recently used item
            lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            del self.cache[lru_key]
            del self.access_times[lru_key]
            logger.info(f"Evicted LRU cache item: {lru_key}")
    
    def get(self, query: str, filters: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get cached result for query
        
        Args:
            query: Search query
            filters: Optional filter string
            
        Returns:
            Cached result or None if not found/expired
        """
        self._cleanup_expired()
        
        key = self._create_key(query, filters)
        
        if key in self.cache:
            self.access_times[key] = time.time()
            logger.info(f"Cache hit for query: {query[:50]}...")
            return self.cache[key]["data"]
        
        logger.info(f"Cache miss for query: {query[:50]}...")
        return None
    
    def set(self, query: str, filters: Optional[str], result: Dict[str, Any]):
        """
        Cache query result
        
        Args:
            query: Search query
            filters: Optional filter string
            result: Query result to cache
        """
        self._cleanup_expired()
        self._evict_lru()
        
        key = self._create_key(query, filters)
        
        self.cache[key] = {
            "data": result,
            "timestamp": time.time()
        }
        self.access_times[key] = time.time()
        
        logger.info(f"Cached result for query: {query[:50]}...")
    
    def clear(self):
        """Clear all cached items"""
        self.cache.clear()
        self.access_times.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        current_time = time.time()
        
        # Count expired items
        expired_count = sum(1 for item in self.cache.values() 
                          if current_time - item["timestamp"] > self.ttl_seconds)
        
        return {
            "total_items": len(self.cache),
            "expired_items": expired_count,
            "active_items": len(self.cache) - expired_count,
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds
        }
    
    def get_history(self) -> List[str]:
        """Get list of recent queries from cache"""
        queries = []
        for key, item in self.cache.items():
            # Try to extract original query from cache data
            if "query" in item["data"]:
                queries.append(item["data"]["query"])
        
        return list(set(queries))  # Remove duplicates


class ResultFormatter:
    """
    Format query results for display
    """
    
    @staticmethod
    def format_answer_for_display(result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format query result for Streamlit display
        
        Args:
            result: Raw query result
            
        Returns:
            Formatted result for display
        """
        formatted = {
            "query": result.get("query", ""),
            "answer": result.get("answer", ""),
            "processing_time": result.get("processing_time", 0),
            "success": result.get("success", False),
            "sources": []
        }
        
        # Format sources
        sources = result.get("sources", [])
        for i, source in enumerate(sources):
            formatted_source = {
                "rank": i + 1,
                "relevance_score": source.get("relevance_score", 0),
                "text": source.get("text", ""),
                "metadata": {
                    "title": source.get("metadata", {}).get("title", "Unknown"),
                    "authors": source.get("metadata", {}).get("authors", []),
                    "page": source.get("metadata", {}).get("page_number", 0),
                    "section": source.get("metadata", {}).get("section_title", "Unknown")
                }
            }
            formatted["sources"].append(formatted_source)
        
        # Add statistics
        formatted["stats"] = {
            "search_results": result.get("search_results_count", 0),
            "reranked_results": result.get("reranked_results_count", 0),
            "source_count": len(formatted["sources"]),
            "has_enhanced_query": "enhanced_queries" in result
        }
        
        if "enhanced_queries" in result:
            formatted["enhanced_queries"] = result["enhanced_queries"]
        
        if "error" in result:
            formatted["error"] = result["error"]
        
        return formatted
    
    @staticmethod
    def format_sources_for_citation(sources: List[Dict[str, Any]]) -> str:
        """
        Format sources for citation display
        
        Args:
            sources: List of source dictionaries
            
        Returns:
            Formatted citation string
        """
        citations = []
        
        for i, source in enumerate(sources):
            metadata = source.get("metadata", {})
            
            # Format author names
            authors = metadata.get("authors", [])
            if authors:
                if isinstance(authors, list):
                    author_names = []
                    for author in authors:
                        if isinstance(author, dict):
                            author_names.append(author.get("name", "Unknown"))
                        else:
                            author_names.append(str(author))
                    author_str = ", ".join(author_names)
                else:
                    author_str = str(authors)
            else:
                author_str = "Unknown Author"
            
            # Format citation
            title = metadata.get("title", "Unknown Title")
            page = metadata.get("page", 0)
            
            citation = f"[{i+1}] {author_str}. {title}"
            if page:
                citation += f", стр. {page}"
            
            citations.append(citation)
        
        return "\n".join(citations)


class TextProcessor:
    """
    Text processing utilities
    """
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
        """
        Truncate text to maximum length
        
        Args:
            text: Text to truncate
            max_length: Maximum length
            suffix: Suffix to add if truncated
            
        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text
        
        # Try to truncate at word boundary
        truncated = text[:max_length - len(suffix)]
        last_space = truncated.rfind(' ')
        
        if last_space > max_length // 2:  # Only use word boundary if not too short
            truncated = truncated[:last_space]
        
        return truncated + suffix
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text for display
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        cleaned = ' '.join(text.split())
        
        # Remove control characters
        cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\n\t')
        
        return cleaned.strip()
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text (simple implementation)
        
        Args:
            text: Text to extract keywords from
            max_keywords: Maximum number of keywords
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction based on word frequency
        words = text.lower().split()
        
        # Filter out common words
        stop_words = {'и', 'в', 'на', 'с', 'по', 'для', 'от', 'до', 'при', 'это', 'как', 'что', 'то', 'все', 'еще', 'уже', 'или', 'но', 'да', 'нет', 'не', 'а', 'о', 'у', 'я', 'он', 'она', 'мы', 'вы', 'они'}
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            word = word.strip('.,!?;:"()[]{}')
            if len(word) > 2 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in keywords[:max_keywords]]


class FileValidator:
    """
    File validation utilities
    """
    
    @staticmethod
    def validate_pdf(file_content: bytes) -> bool:
        """
        Validate if file content is a valid PDF
        
        Args:
            file_content: File content as bytes
            
        Returns:
            True if valid PDF, False otherwise
        """
        if not file_content:
            return False
        
        # Check PDF magic number
        return file_content.startswith(b'%PDF-')
    
    @staticmethod
    def get_file_size_mb(file_content: bytes) -> float:
        """
        Get file size in megabytes
        
        Args:
            file_content: File content as bytes
            
        Returns:
            File size in MB
        """
        return len(file_content) / (1024 * 1024)
    
    @staticmethod
    def validate_file_size(file_content: bytes, max_size_mb: float = 50) -> bool:
        """
        Validate file size
        
        Args:
            file_content: File content as bytes
            max_size_mb: Maximum size in MB
            
        Returns:
            True if size is acceptable, False otherwise
        """
        size_mb = FileValidator.get_file_size_mb(file_content)
        return size_mb <= max_size_mb


class ConfigValidator:
    """
    Configuration validation utilities
    """
    
    @staticmethod
    def validate_api_keys() -> Dict[str, bool]:
        """
        Validate that required API keys are present
        
        Returns:
            Dictionary with validation results for each API key
        """
        from .config import settings
        
        keys_to_check = {
            "openai": settings.openai_api_key,
            "pinecone": settings.pinecone_api_key,
            "cohere": settings.cohere_api_key,
            "llama_cloud": settings.llama_cloud_api_key
        }
        
        validation_results = {}
        for key_name, key_value in keys_to_check.items():
            is_valid = (
                key_value and 
                not key_value.startswith("your_") and 
                len(key_value) > 10
            )
            validation_results[key_name] = is_valid
        
        return validation_results
    
    @staticmethod
    def get_missing_api_keys() -> List[str]:
        """
        Get list of missing or invalid API keys
        
        Returns:
            List of missing API key names
        """
        validation_results = ConfigValidator.validate_api_keys()
        return [key for key, is_valid in validation_results.items() if not is_valid]


class MetricsCollector:
    """
    Simple metrics collection for monitoring system performance
    """
    
    def __init__(self):
        """Initialize metrics collector"""
        self.metrics: Dict[str, List[float]] = {}
        self.counters: Dict[str, int] = {}
    
    def record_timing(self, operation: str, duration: float):
        """
        Record timing for an operation
        
        Args:
            operation: Operation name
            duration: Duration in seconds
        """
        if operation not in self.metrics:
            self.metrics[operation] = []
        
        self.metrics[operation].append(duration)
        
        # Keep only last 100 measurements
        if len(self.metrics[operation]) > 100:
            self.metrics[operation] = self.metrics[operation][-100:]
    
    def increment_counter(self, counter: str):
        """
        Increment a counter
        
        Args:
            counter: Counter name
        """
        self.counters[counter] = self.counters.get(counter, 0) + 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get collected statistics
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "counters": self.counters.copy(),
            "timings": {}
        }
        
        for operation, timings in self.metrics.items():
            if timings:
                stats["timings"][operation] = {
                    "count": len(timings),
                    "avg": sum(timings) / len(timings),
                    "min": min(timings),
                    "max": max(timings),
                    "recent": timings[-10:] if len(timings) >= 10 else timings
                }
        
        return stats
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.counters.clear()


# Global metrics collector instance
metrics = MetricsCollector()


def format_processing_time(seconds: float) -> str:
    """
    Format processing time for display
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"


def create_metadata_filter_from_ui(**kwargs) -> Dict[str, Any]:
    """
    Create metadata filter from UI parameters
    
    Args:
        **kwargs: UI filter parameters
        
    Returns:
        Metadata filter dictionary
    """
    from .vector_store import VectorStore
    
    # Create a temporary vector store to use its filter creation method
    vector_store = VectorStore()
    return vector_store.create_metadata_filter(**kwargs)


def safe_json_serialize(obj: Any) -> str:
    """
    Safely serialize object to JSON, handling non-serializable objects
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON string
    """
    def json_handler(obj):
        if hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):  # custom objects
            return obj.__dict__
        else:
            return str(obj)
    
    try:
        return json.dumps(obj, default=json_handler, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error serializing to JSON: {str(e)}")
        return f"{{\"error\": \"Serialization failed: {str(e)}\"}}"