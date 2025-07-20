"""
Logging configuration for the ArXiv RAG System

This module provides comprehensive logging setup for monitoring system operations,
API calls, and performance metrics as specified in the PRD.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
import json

from .config import settings


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter for structured logging with JSON output
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        
        # Basic log data
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        # Add API call information if present
        if hasattr(record, 'api_call'):
            log_data["api_call"] = record.api_call
        
        # Add performance metrics if present
        if hasattr(record, 'metrics'):
            log_data["metrics"] = record.metrics
        
        return json.dumps(log_data, ensure_ascii=False)


class APICallLogger:
    """
    Logger for API calls with request/response tracking
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize API call logger
        
        Args:
            logger: Base logger to use
        """
        self.logger = logger
    
    def log_api_call(self, service: str, operation: str, 
                    request_data: Optional[Dict[str, Any]] = None,
                    response_data: Optional[Dict[str, Any]] = None,
                    duration: Optional[float] = None,
                    success: bool = True,
                    error: Optional[str] = None):
        """
        Log API call with detailed information
        
        Args:
            service: API service name (openai, pinecone, cohere, llamacloud)
            operation: Operation type (embed, search, rerank, parse, extract)
            request_data: Request parameters (sanitized)
            response_data: Response data (sanitized)
            duration: Call duration in seconds
            success: Whether call was successful
            error: Error message if failed
        """
        
        api_call_data = {
            "service": service,
            "operation": operation,
            "success": success,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if request_data:
            # Sanitize request data (remove sensitive info)
            api_call_data["request"] = self._sanitize_data(request_data)
        
        if response_data:
            # Sanitize response data
            api_call_data["response"] = self._sanitize_data(response_data)
        
        if duration is not None:
            api_call_data["duration_seconds"] = round(duration, 3)
        
        if error:
            api_call_data["error"] = str(error)
        
        # Log with appropriate level
        level = logging.INFO if success else logging.ERROR
        
        # Create log record with extra data
        extra = {"api_call": api_call_data}
        self.logger.log(level, f"API call: {service}.{operation}", extra=extra)
    
    def _sanitize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize data by removing sensitive information
        
        Args:
            data: Data to sanitize
            
        Returns:
            Sanitized data
        """
        if not isinstance(data, dict):
            return {"type": type(data).__name__, "length": len(str(data))}
        
        sanitized = {}
        
        for key, value in data.items():
            # Skip sensitive keys
            if any(sensitive in key.lower() for sensitive in ['key', 'token', 'password', 'secret']):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, str) and len(value) > 1000:
                # Truncate long strings
                sanitized[key] = value[:100] + f"... [truncated, total length: {len(value)}]"
            elif isinstance(value, list) and len(value) > 10:
                # Limit list size
                sanitized[key] = value[:5] + [f"... [truncated, total items: {len(value)}]"]
            elif isinstance(value, dict):
                # Recursively sanitize nested dictionaries
                sanitized[key] = self._sanitize_data(value)
            else:
                sanitized[key] = value
        
        return sanitized


class PerformanceLogger:
    """
    Logger for performance metrics and system monitoring
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize performance logger
        
        Args:
            logger: Base logger to use
        """
        self.logger = logger
    
    def log_performance(self, operation: str, duration: float,
                       input_size: Optional[int] = None,
                       output_size: Optional[int] = None,
                       memory_usage: Optional[int] = None,
                       additional_metrics: Optional[Dict[str, Any]] = None):
        """
        Log performance metrics
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            input_size: Input size (characters, tokens, etc.)
            output_size: Output size
            memory_usage: Memory usage in bytes
            additional_metrics: Additional metrics to log
        """
        
        metrics = {
            "operation": operation,
            "duration_seconds": round(duration, 3),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if input_size is not None:
            metrics["input_size"] = input_size
        
        if output_size is not None:
            metrics["output_size"] = output_size
        
        if memory_usage is not None:
            metrics["memory_usage_bytes"] = memory_usage
        
        if additional_metrics:
            metrics.update(additional_metrics)
        
        # Create log record with metrics
        extra = {"metrics": metrics}
        self.logger.info(f"Performance: {operation}", extra=extra)


def setup_logging(log_level: str = None, log_file: str = None) -> Dict[str, logging.Logger]:
    """
    Set up comprehensive logging for the RAG system
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        
    Returns:
        Dictionary of configured loggers
    """
    
    # Determine log level
    if log_level is None:
        log_level = getattr(settings, 'log_level', 'INFO')
    
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logs directory if using file logging
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    else:
        # Default log file location
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"arxiv_rag_{datetime.now().strftime('%Y%m%d')}.log")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    # Console formatter (human-readable)
    console_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler with structured logging
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(numeric_level)
    
    # Structured formatter for file logging
    structured_formatter = StructuredFormatter()
    file_handler.setFormatter(structured_formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Create specialized loggers
    loggers = {}
    
    # Main application logger
    loggers['app'] = logging.getLogger('arxiv_rag.app')
    
    # Component loggers
    loggers['document_processor'] = logging.getLogger('arxiv_rag.document_processor')
    loggers['metadata_extractor'] = logging.getLogger('arxiv_rag.metadata_extractor')
    loggers['chunking'] = logging.getLogger('arxiv_rag.chunking')
    loggers['vector_store'] = logging.getLogger('arxiv_rag.vector_store')
    loggers['query_engine'] = logging.getLogger('arxiv_rag.query_engine')
    
    # API loggers
    loggers['api.openai'] = logging.getLogger('arxiv_rag.api.openai')
    loggers['api.pinecone'] = logging.getLogger('arxiv_rag.api.pinecone')
    loggers['api.cohere'] = logging.getLogger('arxiv_rag.api.cohere')
    loggers['api.llamacloud'] = logging.getLogger('arxiv_rag.api.llamacloud')
    
    # Performance logger
    loggers['performance'] = logging.getLogger('arxiv_rag.performance')
    
    # Security logger
    loggers['security'] = logging.getLogger('arxiv_rag.security')
    
    # Set up API call logging
    for service in ['openai', 'pinecone', 'cohere', 'llamacloud']:
        api_logger = loggers[f'api.{service}']
        api_logger.api_call_logger = APICallLogger(api_logger)
    
    # Set up performance logging
    loggers['performance'].performance_logger = PerformanceLogger(loggers['performance'])
    
    # Log successful setup
    loggers['app'].info(f"Logging configured: level={log_level}, file={log_file}")
    
    return loggers


def get_logger(name: str) -> logging.Logger:
    """
    Get logger by name
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f'arxiv_rag.{name}')


def log_api_call(service: str, operation: str, **kwargs):
    """
    Convenience function to log API calls
    
    Args:
        service: API service name
        operation: Operation type
        **kwargs: Additional logging parameters
    """
    logger = get_logger(f'api.{service}')
    if hasattr(logger, 'api_call_logger'):
        logger.api_call_logger.log_api_call(service, operation, **kwargs)


def log_performance(operation: str, duration: float, **kwargs):
    """
    Convenience function to log performance metrics
    
    Args:
        operation: Operation name
        duration: Duration in seconds
        **kwargs: Additional metrics
    """
    logger = get_logger('performance')
    if hasattr(logger, 'performance_logger'):
        logger.performance_logger.log_performance(operation, duration, **kwargs)


def log_security_event(event_type: str, details: Dict[str, Any], severity: str = "INFO"):
    """
    Log security-related events
    
    Args:
        event_type: Type of security event
        details: Event details
        severity: Event severity (INFO, WARNING, ERROR)
    """
    logger = get_logger('security')
    
    security_data = {
        "event_type": event_type,
        "details": details,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    extra = {"security_event": security_data}
    
    level = getattr(logging, severity.upper(), logging.INFO)
    logger.log(level, f"Security event: {event_type}", extra=extra)


# Context managers for automatic logging
class LoggedOperation:
    """Context manager for logging operations with automatic timing"""
    
    def __init__(self, operation_name: str, logger_name: str = 'app'):
        """
        Initialize logged operation
        
        Args:
            operation_name: Name of the operation
            logger_name: Logger to use
        """
        self.operation_name = operation_name
        self.logger = get_logger(logger_name)
        self.start_time = None
    
    def __enter__(self):
        """Start timing the operation"""
        self.start_time = datetime.now()
        self.logger.info(f"Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log results"""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(f"Completed operation: {self.operation_name} in {duration:.3f}s")
            log_performance(self.operation_name, duration)
        else:
            self.logger.error(f"Failed operation: {self.operation_name} after {duration:.3f}s", 
                            exc_info=(exc_type, exc_val, exc_tb))


class LoggedAPICall:
    """Context manager for logging API calls"""
    
    def __init__(self, service: str, operation: str, request_data: Optional[Dict[str, Any]] = None):
        """
        Initialize logged API call
        
        Args:
            service: API service name
            operation: Operation type
            request_data: Request data to log
        """
        self.service = service
        self.operation = operation
        self.request_data = request_data
        self.start_time = None
        self.logger = get_logger(f'api.{service}')
    
    def __enter__(self):
        """Start timing the API call"""
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log API call"""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        success = exc_type is None
        error = str(exc_val) if exc_val else None
        
        log_api_call(
            service=self.service,
            operation=self.operation,
            request_data=self.request_data,
            duration=duration,
            success=success,
            error=error
        )


# Initialize logging on module import
_loggers = None

def initialize_logging():
    """Initialize logging system"""
    global _loggers
    if _loggers is None:
        _loggers = setup_logging()
    return _loggers

# Auto-initialize with default settings
try:
    initialize_logging()
except Exception as e:
    print(f"Warning: Failed to initialize logging: {e}")
    # Fallback to basic logging
    logging.basicConfig(level=logging.INFO)