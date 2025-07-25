"""
Simple logging configuration for RAG system batch processing

This module provides basic logging setup for tracking batch document processing
with minimal overhead and clear visibility into system operations.
"""

import logging
import sys
from typing import Optional


class SimpleFormatter(logging.Formatter):
    """Simple formatter for clean log output"""
    
    def format(self, record):
        # Format: YYYY-MM-DD HH:MM:SS LEVEL Message
        return f"{self.formatTime(record, '%Y-%m-%d %H:%M:%S')} {record.levelname} {record.getMessage()}"


def setup_batch_logger(name: str = "batch_processing", level: str = "INFO") -> logging.Logger:
    """
    Setup simple logger for batch processing
    
    Args:
        name: Logger name
        level: Logging level (INFO, DEBUG, etc.)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Set level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    
    # Set formatter
    formatter = SimpleFormatter()
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def setup_chunking_logger(name: str = "chunking", level: str = "DEBUG") -> logging.Logger:
    """
    Setup logger specifically for chunking details
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    return setup_batch_logger(name, level)


# Global logger instances for easy access
batch_logger = setup_batch_logger("batch_processing", "INFO")
chunking_logger = setup_chunking_logger("chunking", "DEBUG")
document_logger = setup_batch_logger("document_processing", "INFO")
vector_logger = setup_batch_logger("vector_store", "INFO")


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Get or create a logger with the specified configuration
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    return setup_batch_logger(name, level)


def set_logging_level(level: str):
    """
    Set logging level for all batch processing loggers
    
    Args:
        level: New logging level (INFO, DEBUG, WARNING, ERROR)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Update all known loggers
    for logger_name in ["batch_processing", "chunking", "document_processing", "vector_store"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        for handler in logger.handlers:
            handler.setLevel(log_level)


def log_batch_start(total_files: int, config: Optional[dict] = None):
    """Log batch processing start"""
    config_str = f", config: {config}" if config else ""
    batch_logger.info(f"Batch started: {total_files} files{config_str}")


def log_batch_complete(successful: int, failed: int):
    """Log batch processing completion"""
    batch_logger.info(f"Batch completed: {successful} success, {failed} failed")


def log_document_start(filename: str):
    """Log document processing start"""
    document_logger.info(f"Processing started: {filename}")


def log_document_stage(filename: str, stage: str, extra_info: str = ""):
    """Log document processing stage completion"""
    info = f" - {extra_info}" if extra_info else ""
    document_logger.info(f"{stage.title()} completed: {filename}{info}")


def log_document_error(filename: str, stage: str, error: str):
    """Log document processing error"""
    document_logger.error(f"Failed at {stage}: {filename} - {error}")


def log_chunks_created(filename: str, chunk_ids: list):
    """Log created chunk IDs for a document"""
    chunking_logger.info(f"Document chunked: {filename}")
    chunking_logger.debug(f"Chunks created: {chunk_ids}")


def log_chunks_indexed(filename: str, count: int):
    """Log successful chunk indexing"""
    vector_logger.info(f"Indexed {count} chunks for {filename}")