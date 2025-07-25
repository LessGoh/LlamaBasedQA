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
from typing import Optional, Dict, Any, List
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
    
    # Parallel processing loggers
    loggers['workflow'] = logging.getLogger('arxiv_rag.workflow')
    loggers['parallel'] = logging.getLogger('arxiv_rag.parallel')
    loggers['batch'] = logging.getLogger('arxiv_rag.batch')
    loggers['queue'] = logging.getLogger('arxiv_rag.queue')
    loggers['dispatcher'] = logging.getLogger('arxiv_rag.dispatcher')
    loggers['progress'] = logging.getLogger('arxiv_rag.progress')
    
    # Error handling and monitoring loggers
    loggers['error_handler'] = logging.getLogger('arxiv_rag.error_handler')
    loggers['resource_monitor'] = logging.getLogger('arxiv_rag.resource_monitor')
    
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

class BatchProcessingLogger:
    """
    Specialized logger for batch processing operations
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_batch_start(self, batch_id: str, total_documents: int, config: Dict[str, Any]):
        """Log batch processing start"""
        self.logger.info(
            f"Batch processing started: {batch_id}",
            extra={
                "batch_event": {
                    "event": "batch_start",
                    "batch_id": batch_id,
                    "total_documents": total_documents,
                    "config": config,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        )
    
    def log_batch_progress(self, batch_id: str, completed: int, failed: int, 
                          active: int, remaining: int):
        """Log batch processing progress"""
        self.logger.info(
            f"Batch progress: {batch_id} ({completed}/{completed+failed+remaining} completed)",
            extra={
                "batch_event": {
                    "event": "batch_progress",
                    "batch_id": batch_id,
                    "completed": completed,
                    "failed": failed,
                    "active": active,
                    "remaining": remaining,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        )
    
    def log_batch_complete(self, batch_id: str, results: Dict[str, Any]):
        """Log batch processing completion"""
        self.logger.info(
            f"Batch processing completed: {batch_id}",
            extra={
                "batch_event": {
                    "event": "batch_complete",
                    "batch_id": batch_id,
                    "results": results,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        )


class WorkflowLogger:
    """
    Specialized logger for workflow operations
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_workflow_start(self, workflow_id: str, document_id: str, filename: str):
        """Log workflow start"""
        self.logger.info(
            f"Workflow started: {workflow_id} for {filename}",
            extra={
                "workflow_event": {
                    "event": "workflow_start",
                    "workflow_id": workflow_id,
                    "document_id": document_id,
                    "filename": filename,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        )
    
    def log_workflow_stage(self, workflow_id: str, document_id: str, stage: str, 
                          status: str, duration: Optional[float] = None):
        """Log workflow stage completion"""
        self.logger.info(
            f"Workflow stage: {workflow_id} - {stage} ({status})",
            extra={
                "workflow_event": {
                    "event": "workflow_stage",
                    "workflow_id": workflow_id,
                    "document_id": document_id,
                    "stage": stage,
                    "status": status,
                    "duration": duration,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        )
    
    def log_workflow_complete(self, workflow_id: str, document_id: str, 
                             success: bool, total_duration: float, 
                             error: Optional[str] = None):
        """Log workflow completion"""
        level = logging.INFO if success else logging.ERROR
        message = f"Workflow completed: {workflow_id}" if success else f"Workflow failed: {workflow_id}"
        
        self.logger.log(
            level,
            message,
            extra={
                "workflow_event": {
                    "event": "workflow_complete",
                    "workflow_id": workflow_id,
                    "document_id": document_id,
                    "success": success,
                    "total_duration": total_duration,
                    "error": error,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        )


class ResourceMonitorLogger:
    """
    Logger for resource monitoring and system health
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_resource_usage(self, memory_mb: float, disk_mb: float, 
                          cpu_percent: float, active_workflows: int):
        """Log current resource usage"""
        self.logger.info(
            f"Resource usage: Memory={memory_mb:.1f}MB, Disk={disk_mb:.1f}MB, CPU={cpu_percent:.1f}%",
            extra={
                "resource_event": {
                    "event": "resource_usage",
                    "memory_mb": memory_mb,
                    "disk_mb": disk_mb,
                    "cpu_percent": cpu_percent,
                    "active_workflows": active_workflows,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        )
    
    def log_resource_warning(self, resource_type: str, current_value: float, 
                           threshold: float, message: str):
        """Log resource threshold warnings"""
        self.logger.warning(
            f"Resource warning: {resource_type} - {message}",
            extra={
                "resource_event": {
                    "event": "resource_warning",
                    "resource_type": resource_type,
                    "current_value": current_value,
                    "threshold": threshold,
                    "message": message,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        )
    
    def log_resource_critical(self, resource_type: str, current_value: float, 
                             limit: float, action: str):
        """Log critical resource conditions"""
        self.logger.critical(
            f"Resource critical: {resource_type} - taking action: {action}",
            extra={
                "resource_event": {
                    "event": "resource_critical",
                    "resource_type": resource_type,
                    "current_value": current_value,
                    "limit": limit,
                    "action": action,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        )


class ErrorAnalysisLogger:
    """
    Logger for error analysis and troubleshooting
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_error_pattern(self, error_type: str, frequency: int, 
                         last_occurrence: str, affected_documents: List[str]):
        """Log detected error patterns"""
        self.logger.warning(
            f"Error pattern detected: {error_type} (frequency: {frequency})",
            extra={
                "error_analysis": {
                    "event": "error_pattern",
                    "error_type": error_type,
                    "frequency": frequency,
                    "last_occurrence": last_occurrence,
                    "affected_documents": affected_documents[:10],  # Limit for log size
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        )
    
    def log_recovery_action(self, document_id: str, error_type: str, 
                           action: str, success: bool):
        """Log error recovery actions"""
        level = logging.INFO if success else logging.ERROR
        self.logger.log(
            level,
            f"Recovery action: {action} for {document_id} ({error_type})",
            extra={
                "error_analysis": {
                    "event": "recovery_action",
                    "document_id": document_id,
                    "error_type": error_type,
                    "action": action,
                    "success": success,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        )


def setup_specialized_loggers(loggers: Dict[str, logging.Logger]):
    """
    Set up specialized loggers for parallel processing
    
    Args:
        loggers: Dictionary of existing loggers
    """
    # Add specialized logger instances
    if 'batch' in loggers:
        loggers['batch'].batch_logger = BatchProcessingLogger(loggers['batch'])
    
    if 'workflow' in loggers:
        loggers['workflow'].workflow_logger = WorkflowLogger(loggers['workflow'])
    
    if 'resource_monitor' in loggers:
        loggers['resource_monitor'].resource_logger = ResourceMonitorLogger(loggers['resource_monitor'])
    
    if 'error_handler' in loggers:
        loggers['error_handler'].error_analysis_logger = ErrorAnalysisLogger(loggers['error_handler'])


def get_batch_logger() -> Optional['BatchProcessingLogger']:
    """Get batch processing logger"""
    logger = get_logger('batch')
    return getattr(logger, 'batch_logger', None)


def get_workflow_logger() -> Optional['WorkflowLogger']:
    """Get workflow logger"""
    logger = get_logger('workflow')
    return getattr(logger, 'workflow_logger', None)


def get_resource_logger() -> Optional['ResourceMonitorLogger']:
    """Get resource monitoring logger"""
    logger = get_logger('resource_monitor')
    return getattr(logger, 'resource_logger', None)


def get_error_analysis_logger() -> Optional['ErrorAnalysisLogger']:
    """Get error analysis logger"""
    logger = get_logger('error_handler')
    return getattr(logger, 'error_analysis_logger', None)


def log_batch_event(event_type: str, batch_id: str, **kwargs):
    """Convenience function for batch event logging"""
    batch_logger = get_batch_logger()
    if batch_logger:
        if event_type == "start":
            batch_logger.log_batch_start(batch_id, **kwargs)
        elif event_type == "progress":
            batch_logger.log_batch_progress(batch_id, **kwargs)
        elif event_type == "complete":
            batch_logger.log_batch_complete(batch_id, **kwargs)


def log_workflow_event(event_type: str, workflow_id: str, document_id: str, **kwargs):
    """Convenience function for workflow event logging"""
    workflow_logger = get_workflow_logger()
    if workflow_logger:
        if event_type == "start":
            workflow_logger.log_workflow_start(workflow_id, document_id, **kwargs)
        elif event_type == "stage":
            workflow_logger.log_workflow_stage(workflow_id, document_id, **kwargs)
        elif event_type == "complete":
            workflow_logger.log_workflow_complete(workflow_id, document_id, **kwargs)


def configure_parallel_logging():
    """Configure logging specifically for parallel processing"""
    global _loggers
    if _loggers is None:
        _loggers = setup_logging()
    
    # Set up specialized loggers
    setup_specialized_loggers(_loggers)
    
    return _loggers


class AsyncLogger:
    """
    Asynchronous logger wrapper for high-throughput logging
    """
    
    def __init__(self, logger: logging.Logger, queue_size: int = 1000):
        self.logger = logger
        self.queue_size = queue_size
        self._queue = None
        self._handler = None
    
    async def async_log(self, level: int, message: str, **kwargs):
        """Log message asynchronously"""
        # This is a simplified implementation
        # In a real scenario, you might use asyncio.Queue and a background task
        self.logger.log(level, message, **kwargs)
    
    async def async_info(self, message: str, **kwargs):
        """Log info message asynchronously"""
        await self.async_log(logging.INFO, message, **kwargs)
    
    async def async_error(self, message: str, **kwargs):
        """Log error message asynchronously"""
        await self.async_log(logging.ERROR, message, **kwargs)
    
    async def async_warning(self, message: str, **kwargs):
        """Log warning message asynchronously"""
        await self.async_log(logging.WARNING, message, **kwargs)


def get_async_logger(name: str) -> AsyncLogger:
    """Get asynchronous logger wrapper"""
    logger = get_logger(name)
    return AsyncLogger(logger)


# Enhanced initialization
def initialize_parallel_logging():
    """Initialize logging system with parallel processing support"""
    global _loggers
    if _loggers is None:
        _loggers = setup_logging()
        setup_specialized_loggers(_loggers)
    return _loggers


# Auto-initialize with default settings
try:
    initialize_parallel_logging()
except Exception as e:
    print(f"Warning: Failed to initialize logging: {e}")
    # Fallback to basic logging
    logging.basicConfig(level=logging.INFO)