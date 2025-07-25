"""
Utility functions and helpers for the RAG system

This module provides utility functions used across the RAG system including
caching, formatting, validation, and helper classes.
"""

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import pickle



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
            pass  # Expired keys have been removed
    
    def _evict_lru(self):
        """Evict least recently used items if cache is full"""
        while len(self.cache) >= self.max_size:
            # Find least recently used item
            lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            del self.cache[lru_key]
            del self.access_times[lru_key]
    
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
            return self.cache[key]["data"]
        
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
        
    
    def clear(self):
        """Clear all cached items"""
        self.cache.clear()
        self.access_times.clear()
    
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
        return f"{{\"error\": \"Serialization failed: {str(e)}\"}}"


class AsyncTaskManager:
    """
    Utility for managing asynchronous tasks in parallel processing
    """
    
    def __init__(self, max_concurrent_tasks: int = 10):
        """
        Initialize async task manager
        
        Args:
            max_concurrent_tasks: Maximum number of concurrent tasks
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.active_tasks: Dict[str, Any] = {}
        self.completed_tasks: Dict[str, Any] = {}
        self.failed_tasks: Dict[str, Any] = {}
    
    async def run_with_semaphore(self, semaphore, task_func, task_id: str, *args, **kwargs):
        """
        Run task with semaphore control
        
        Args:
            semaphore: asyncio.Semaphore for concurrency control
            task_func: Async function to execute
            task_id: Unique task identifier
            *args: Task function arguments
            **kwargs: Task function keyword arguments
        """
        async with semaphore:
            try:
                self.active_tasks[task_id] = {
                    "start_time": time.time(),
                    "status": "running"
                }
                
                result = await task_func(*args, **kwargs)
                
                self.completed_tasks[task_id] = {
                    "result": result,
                    "completion_time": time.time(),
                    "duration": time.time() - self.active_tasks[task_id]["start_time"]
                }
                
                del self.active_tasks[task_id]
                return result
                
            except Exception as e:
                self.failed_tasks[task_id] = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "failure_time": time.time(),
                    "duration": time.time() - self.active_tasks[task_id]["start_time"]
                }
                
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
                
                raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get current task manager status"""
        return {
            "active_count": len(self.active_tasks),
            "completed_count": len(self.completed_tasks),
            "failed_count": len(self.failed_tasks),
            "active_tasks": list(self.active_tasks.keys()),
            "max_concurrent": self.max_concurrent_tasks
        }
    
    def cleanup(self):
        """Clean up task data"""
        self.active_tasks.clear()
        self.completed_tasks.clear()
        self.failed_tasks.clear()


class ResourceMonitor:
    """
    System resource monitoring utilities
    """
    
    def __init__(self):
        self.history_size = 100
        self.memory_history = []
        self.cpu_history = []
        self.disk_history = []
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            return 0.0
    
    def get_disk_usage(self, path: str = "/") -> Dict[str, float]:
        """Get disk usage for specified path"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(path)
            return {
                "total_gb": total / (1024**3),
                "used_gb": used / (1024**3),
                "free_gb": free / (1024**3),
                "usage_percent": (used / total) * 100
            }
        except Exception:
            return {"total_gb": 0, "used_gb": 0, "free_gb": 0, "usage_percent": 0}
    
    def record_usage(self):
        """Record current resource usage"""
        self.memory_history.append(self.get_memory_usage())
        self.cpu_history.append(self.get_cpu_usage())
        
        # Keep only recent history
        for history in [self.memory_history, self.cpu_history]:
            if len(history) > self.history_size:
                history.pop(0)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics"""
        stats = {
            "current": {
                "memory_mb": self.get_memory_usage(),
                "cpu_percent": self.get_cpu_usage(),
                "disk": self.get_disk_usage()
            }
        }
        
        if self.memory_history:
            stats["memory_stats"] = {
                "avg": sum(self.memory_history) / len(self.memory_history),
                "max": max(self.memory_history),
                "min": min(self.memory_history)
            }
        
        if self.cpu_history:
            stats["cpu_stats"] = {
                "avg": sum(self.cpu_history) / len(self.cpu_history),
                "max": max(self.cpu_history),
                "min": min(self.cpu_history)
            }
        
        return stats
    
    def check_resource_limits(self, memory_limit_mb: float = 4096, 
                            cpu_limit_percent: float = 90) -> Dict[str, Any]:
        """Check if resource usage exceeds limits"""
        current_memory = self.get_memory_usage()
        current_cpu = self.get_cpu_usage()
        
        warnings = []
        critical = []
        
        if current_memory > memory_limit_mb * 0.8:
            warnings.append(f"Memory usage high: {current_memory:.1f}MB / {memory_limit_mb}MB")
        
        if current_memory > memory_limit_mb:
            critical.append(f"Memory limit exceeded: {current_memory:.1f}MB / {memory_limit_mb}MB")
        
        if current_cpu > cpu_limit_percent * 0.8:
            warnings.append(f"CPU usage high: {current_cpu:.1f}%")
        
        if current_cpu > cpu_limit_percent:
            critical.append(f"CPU limit exceeded: {current_cpu:.1f}%")
        
        return {
            "warnings": warnings,
            "critical": critical,
            "within_limits": len(critical) == 0
        }


class BatchResultAggregator:
    """
    Utility for aggregating results from batch processing
    """
    
    def __init__(self):
        self.successful_results = []
        self.failed_results = []
        self.processing_stats = {}
    
    def add_success(self, document_id: str, result: Dict[str, Any]):
        """Add successful processing result"""
        self.successful_results.append({
            "document_id": document_id,
            "result": result,
            "timestamp": time.time()
        })
    
    def add_failure(self, document_id: str, error: str, error_type: str = "Unknown"):
        """Add failed processing result"""
        self.failed_results.append({
            "document_id": document_id,
            "error": error,
            "error_type": error_type,
            "timestamp": time.time()
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get processing summary"""
        total_processed = len(self.successful_results) + len(self.failed_results)
        success_rate = (len(self.successful_results) / max(total_processed, 1)) * 100
        
        # Analyze error patterns
        error_types = {}
        for failure in self.failed_results:
            error_type = failure["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_processed": total_processed,
            "successful": len(self.successful_results),
            "failed": len(self.failed_results),
            "success_rate": success_rate,
            "error_types": error_types,
            "processing_stats": self.processing_stats
        }
    
    def get_detailed_report(self) -> Dict[str, Any]:
        """Get detailed processing report"""
        return {
            "summary": self.get_summary(),
            "successful_results": self.successful_results,
            "failed_results": self.failed_results,
            "generated_at": datetime.now().isoformat()
        }


class ProgressCalculator:
    """
    Utility for calculating and estimating progress
    """
    
    @staticmethod
    def calculate_completion_percentage(completed: int, total: int) -> float:
        """Calculate completion percentage"""
        return (completed / max(total, 1)) * 100
    
    @staticmethod
    def estimate_remaining_time(completed: int, total: int, elapsed_time: float) -> Optional[float]:
        """Estimate remaining processing time"""
        if completed <= 0:
            return None
        
        remaining = total - completed
        if remaining <= 0:
            return 0.0
        
        avg_time_per_item = elapsed_time / completed
        return remaining * avg_time_per_item
    
    @staticmethod
    def calculate_throughput(processed: int, elapsed_time: float) -> float:
        """Calculate processing throughput (items per second)"""
        if elapsed_time <= 0:
            return 0.0
        return processed / elapsed_time
    
    @staticmethod
    def calculate_eta(remaining_items: int, throughput: float) -> Optional[datetime]:
        """Calculate estimated time of arrival"""
        if throughput <= 0 or remaining_items <= 0:
            return None
        
        remaining_seconds = remaining_items / throughput
        return datetime.now() + timedelta(seconds=remaining_seconds)


class ErrorPatternAnalyzer:
    """
    Utility for analyzing error patterns in batch processing
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.error_history = []
        self.pattern_cache = {}
    
    def record_error(self, document_id: str, error_type: str, error_message: str, 
                    stage: str, timestamp: Optional[float] = None):
        """Record an error occurrence"""
        error_record = {
            "document_id": document_id,
            "error_type": error_type,
            "error_message": error_message,
            "stage": stage,
            "timestamp": timestamp or time.time()
        }
        
        self.error_history.append(error_record)
        
        # Keep history size manageable
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        # Clear pattern cache to force recalculation
        self.pattern_cache.clear()
    
    def analyze_patterns(self, time_window_hours: float = 24) -> Dict[str, Any]:
        """Analyze error patterns within time window"""
        cache_key = f"patterns_{time_window_hours}"
        if cache_key in self.pattern_cache:
            return self.pattern_cache[cache_key]
        
        current_time = time.time()
        window_start = current_time - (time_window_hours * 3600)
        
        # Filter errors within time window
        recent_errors = [
            error for error in self.error_history
            if error["timestamp"] >= window_start
        ]
        
        # Analyze patterns
        patterns = {
            "total_errors": len(recent_errors),
            "error_types": {},
            "stages": {},
            "frequency": {},
            "trending_errors": [],
            "recommendations": []
        }
        
        # Count by error type
        for error in recent_errors:
            error_type = error["error_type"]
            stage = error["stage"]
            
            patterns["error_types"][error_type] = patterns["error_types"].get(error_type, 0) + 1
            patterns["stages"][stage] = patterns["stages"].get(stage, 0) + 1
        
        # Find most common error types
        sorted_errors = sorted(patterns["error_types"].items(), key=lambda x: x[1], reverse=True)
        patterns["most_common"] = sorted_errors[:5]
        
        # Generate recommendations
        if patterns["total_errors"] > 0:
            if sorted_errors:
                most_common_error = sorted_errors[0][0]
                patterns["recommendations"].append(
                    f"Focus on resolving '{most_common_error}' errors which account for "
                    f"{patterns['error_types'][most_common_error]} occurrences"
                )
        
        self.pattern_cache[cache_key] = patterns
        return patterns
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all recorded errors"""
        if not self.error_history:
            return {"message": "No errors recorded"}
        
        total_errors = len(self.error_history)
        unique_error_types = len(set(error["error_type"] for error in self.error_history))
        
        latest_error = max(self.error_history, key=lambda x: x["timestamp"])
        
        return {
            "total_errors": total_errors,
            "unique_error_types": unique_error_types,
            "latest_error": {
                "type": latest_error["error_type"],
                "stage": latest_error["stage"],
                "timestamp": datetime.fromtimestamp(latest_error["timestamp"]).isoformat()
            },
            "error_rate": self._calculate_error_rate()
        }
    
    def _calculate_error_rate(self) -> Dict[str, float]:
        """Calculate error rate over different time periods"""
        current_time = time.time()
        rates = {}
        
        for period_name, hours in [("last_hour", 1), ("last_day", 24), ("last_week", 168)]:
            window_start = current_time - (hours * 3600)
            period_errors = [
                error for error in self.error_history
                if error["timestamp"] >= window_start
            ]
            rates[period_name] = len(period_errors)
        
        return rates


# Global instances for easy access
resource_monitor = ResourceMonitor()
error_analyzer = ErrorPatternAnalyzer()


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information"""
    import platform
    import sys
    
    info = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor()
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable
        },
        "resources": resource_monitor.get_usage_stats()
    }
    
    return info


def create_batch_config_from_settings() -> Dict[str, Any]:
    """Create batch processing config from system settings"""
    try:
        from .config import settings
        return {
            "max_parallel_workflows": settings.max_parallel_workflows,
            "workflow_timeout": settings.workflow_timeout,
            "queue_max_size": settings.queue_max_size,
            "enable_retries": settings.enable_retries,
            "max_retries_per_document": settings.max_retries_per_document
        }
    except ImportError:
        return {
            "max_parallel_workflows": 2,
            "workflow_timeout": 1800.0,
            "queue_max_size": 20,
            "enable_retries": True,
            "max_retries_per_document": 2
        }