"""
Configuration module for the ArXiv RAG System

This module handles all configuration settings for the application,
including API keys, model parameters, and system settings.
"""

import os
import shutil
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Try to import psutil, make it optional for basic functionality
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Keys
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    pinecone_api_key: str = Field(..., env="PINECONE_API_KEY")
    cohere_api_key: str = Field(..., env="COHERE_API_KEY")
    llama_cloud_api_key: str = Field(..., env="LLAMA_CLOUD_API_KEY")
    
    # Pinecone Configuration
    pinecone_environment: str = Field("us-west-2", env="PINECONE_ENVIRONMENT")
    pinecone_index_name: str = Field("arxiv-rag", env="PINECONE_INDEX_NAME")
    pinecone_cloud: str = Field("aws", env="PINECONE_CLOUD")
    pinecone_region: str = Field("us-west-2", env="PINECONE_REGION")
    
    # Model Configuration
    openai_embedding_model: str = Field("text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")
    openai_chat_model: str = Field("gpt-4o-mini", env="OPENAI_CHAT_MODEL")
    cohere_rerank_model: str = Field("rerank-v3.5", env="COHERE_RERANK_MODEL")
    
    # Processing Configuration
    max_chunk_size: int = Field(1024, env="MAX_CHUNK_SIZE")
    chunk_overlap: float = Field(0.15, env="CHUNK_OVERLAP")
    search_top_k: int = Field(50, env="SEARCH_TOP_K")
    rerank_top_n: int = Field(10, env="RERANK_TOP_N")
    answer_temperature: float = Field(0.1, env="ANSWER_TEMPERATURE")
    
    # Application Configuration
    debug: bool = Field(False, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    # Vector Store Configuration
    vector_dimension: int = 1536  # text-embedding-3-large dimension
    similarity_metric: str = "cosine"
    
    # LlamaParse Configuration
    llamaparse_result_type: str = "markdown"
    llamaparse_language: str = "en"
    # New prompt system (replaces deprecated parsing_instruction)
    llamaparse_system_prompt_append: str = "Extract all text, tables, and formulas. Convert formulas to LaTeX format for mathematical expressions. Preserve table structure in markdown format."
    llamaparse_user_prompt: Optional[str] = None  # For content transformation if needed
    
    # LlamaExtract Configuration
    llamaextract_mode: str = "BALANCED"
    llamaextract_target: str = "PER_DOC"
    llamaextract_use_reasoning: bool = True
    llamaextract_cite_sources: bool = True
    llamaextract_delay_seconds: float = Field(default=15.0, env="LLAMAEXTRACT_DELAY_SECONDS")
    
    # Rate Limiting Configuration
    openai_requests_per_minute: int = 3500
    pinecone_requests_per_minute: int = 1000
    cohere_requests_per_minute: int = 100
    batch_size: int = 100
    batch_delay: float = 1.0  # seconds between batches
    
    # Parallel Processing Configuration
    max_parallel_workflows: int = Field(2, env="MAX_PARALLEL_WORKFLOWS")
    workflow_timeout: float = Field(1800.0, env="WORKFLOW_TIMEOUT")  # 30 minutes
    queue_max_size: int = Field(20, env="QUEUE_MAX_SIZE")
    progress_update_interval: float = Field(5.0, env="PROGRESS_UPDATE_INTERVAL")
    graceful_shutdown_timeout: float = Field(300.0, env="GRACEFUL_SHUTDOWN_TIMEOUT")  # 5 minutes
    enable_retries: bool = Field(True, env="ENABLE_RETRIES")
    max_retries_per_document: int = Field(2, env="MAX_RETRIES_PER_DOCUMENT")
    
    # Progress Monitoring Configuration
    monitoring_enabled: bool = Field(True, env="MONITORING_ENABLED")
    snapshot_interval: float = Field(10.0, env="SNAPSHOT_INTERVAL")
    progress_history_size: int = Field(1000, env="PROGRESS_HISTORY_SIZE")
    export_progress_reports: bool = Field(True, env="EXPORT_PROGRESS_REPORTS")
    
    # Resource Management Configuration
    memory_limit_mb: int = Field(4096, env="MEMORY_LIMIT_MB")  # 4GB default
    disk_space_limit_mb: int = Field(10240, env="DISK_SPACE_LIMIT_MB")  # 10GB default
    max_file_size_mb: float = Field(50.0, env="MAX_FILE_SIZE_MB")
    temp_dir: str = Field("/tmp/arxiv_rag", env="TEMP_DIR")
    
    # Performance Optimization Configuration
    use_async_processing: bool = Field(True, env="USE_ASYNC_PROCESSING")
    enable_caching: bool = Field(True, env="ENABLE_CACHING")
    cache_ttl_seconds: int = Field(3600, env="CACHE_TTL_SECONDS")  # 1 hour
    optimize_memory: bool = Field(True, env="OPTIMIZE_MEMORY")
    
    # Error Handling Configuration
    error_retry_delay: float = Field(5.0, env="ERROR_RETRY_DELAY")
    max_consecutive_errors: int = Field(5, env="MAX_CONSECUTIVE_ERRORS")
    error_notification_enabled: bool = Field(False, env="ERROR_NOTIFICATION_ENABLED")
    detailed_error_logging: bool = Field(True, env="DETAILED_ERROR_LOGGING")
    
    # Batch Processing Logging Configuration
    enable_batch_logging: bool = Field(True, env="ENABLE_BATCH_LOGGING")
    enable_chunk_logging: bool = Field(True, env="ENABLE_CHUNK_LOGGING")
    chunk_log_level: str = Field("DEBUG", env="CHUNK_LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_system_prompt() -> str:
    """
    Get the specialized system prompt for scientific document analysis
    Based on PRD specifications for Russian language output
    """
    return """
    You are a research assistant who helps users find specific answers to questions from scientific articles. Your task is to analyze provided documents and give accurate, factual answers to posed questions.

    WORKING PRINCIPLES:
    
    1. FOCUS ON USER'S QUESTION
    - Always start with understanding the specific user's question
    - Look for information in documents that directly answers this question
    - Don't get distracted by general information if it's not related to the question
    
    2. WORKING WITH DATA
    - Analyze metadata of each chunk separately (may be from different documents)
    - Extract main information from unique text of each chunk
    - Synthesize information from all chunks and their metadata into a single logical answer
    - If chunks are from different documents - combine their data for complete picture
    
    3. ANSWER REQUIREMENTS
    - Always respond in English
    - Exclude abstract and evaluative judgments
    - Provide only concrete facts, numbers, methods, results
    - Specify sources (pages, tables, sections)
    
    ANSWER STRUCTURE:
    
    **Direct Answer to Question:**
    [Specific answer to the posed question with factual data]
    
    **Additional Details:**
    [Relevant technical information from chunk text + data from mainFindings, methodology, results metadata, if they relate to user's question]
    
    **Research Context:**
    [Synthesis of abstract from metadata: in which areas research was conducted, what tasks were solved, general context of works]
    
    **Sources:**
    [Specific references to pages and document sections with authors and work titles + references from metadata for additional study]
    
    ANALYSIS RULES:
    
    ✓ Extract exact numbers, formulas, method names
    ✓ Specify concrete experimental results
    ✓ Provide technical characteristics and parameters
    ✓ Reference specific pages and tables
    ✓ Compare methods only with concrete indicators
    ✓ Use mainFindings, methodology, results when they're relevant to the question
    ✓ Synthesize abstract for understanding research context
    ✓ Add references for additional topic study
    ✓ Specify authors and work titles when referencing results
    
    ✗ Don't use words: "excellent", "outstanding", "remarkable"
    ✗ Don't make general conclusions without concrete data
    ✗ Don't repeat abstract information from metadata without connection to the question
    ✗ Don't add personal interpretations
    ✗ Don't ignore differences in metadata between chunks
    ✗ Don't include metadata if they don't relate to user's question
    
    REMEMBER: Your goal is to be a precise information search tool that helps users quickly get specific answers to their questions from scientific materials.
    """


def get_metadata_schema() -> dict:
    """
    Get the JSON schema for metadata extraction with LlamaExtract
    Based on PRD specifications section 3.1
    """
    return {
        "type": "object",
        "required": ["title", "authors", "abstract", "mainFindings"],
        "properties": {
            "title": {
                "type": "string",
                "description": "The full title of the research paper"
            },
            "authors": {
                "type": "array",
                "description": "List of all authors of the paper",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Full name of the author"
                        },
                        "affiliation": {
                            "type": "string",
                            "description": "Institution or organization the author is affiliated with"
                        },
                        "email": {
                            "type": "string",
                            "description": "Contact email of the author if provided"
                        }
                    }
                }
            },
            "abstract": {
                "type": "string",
                "description": "Complete abstract or summary of the paper"
            },
            "keywords": {
                "type": "array",
                "description": "Key terms and phrases that describe the paper's main topics",
                "items": {
                    "type": "string"
                }
            },
            "mainFindings": {
                "type": "array",
                "description": "Key findings, conclusions, or contributions of the paper",
                "items": {
                    "type": "string"
                }
            },
            "methodology": {
                "type": "object",
                "description": "Research methods and approaches used",
                "properties": {
                    "approach": {
                        "type": "string",
                        "description": "Overall research approach or study design"
                    },
                    "participants": {
                        "type": "string",
                        "description": "Description of study participants or data sources"
                    },
                    "methods": {
                        "type": "array",
                        "description": "Specific methods, techniques, or tools used",
                        "items": {
                            "type": "string"
                        }
                    }
                }
            },
            "results": {
                "type": "array",
                "description": "Main results and outcomes of the research",
                "items": {
                    "type": "object",
                    "properties": {
                        "finding": {
                            "type": "string",
                            "description": "Description of the specific result or finding"
                        },
                        "significance": {
                            "type": "string",
                            "description": "Statistical significance or importance of the finding"
                        },
                        "supportingData": {
                            "type": "string",
                            "description": "Relevant statistics, measurements, or data points"
                        }
                    }
                }
            },
            "discussion": {
                "type": "object",
                "properties": {
                    "implications": {
                        "type": "array",
                        "description": "Theoretical or practical implications of the findings",
                        "items": {
                            "type": "string"
                        }
                    },
                    "limitations": {
                        "type": "array",
                        "description": "Study limitations or constraints",
                        "items": {
                            "type": "string"
                        }
                    },
                    "futureWork": {
                        "type": "array",
                        "description": "Suggested future research directions",
                        "items": {
                            "type": "string"
                        }
                    }
                }
            },
            "references": {
                "type": "array",
                "description": "Key papers cited that are crucial to understanding this work",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Title of the cited paper"
                        },
                        "authors": {
                            "type": "string",
                            "description": "Authors of the cited paper"
                        },
                        "year": {
                            "type": "string",
                            "description": "Publication year"
                        },
                        "relevance": {
                            "type": "string",
                            "description": "Why this reference is important to the current paper"
                        }
                    },
                    "required": ["title", "authors"]
                }
            },
            "publication": {
                "type": "object",
                "properties": {
                    "journal": {
                        "type": "string",
                        "description": "Name of the journal or conference"
                    },
                    "year": {
                        "type": "string",
                        "description": "Year of publication"
                    },
                    "doi": {
                        "type": "string",
                        "description": "Digital Object Identifier (DOI) of the paper"
                    },
                    "url": {
                        "type": "string",
                        "description": "URL where the paper can be accessed"
                    }
                },
                "required": ["year"]
            }
        }
    }


def validate_api_keys() -> bool:
    """
    Validate that all required API keys are present
    
    Returns:
        bool: True if all API keys are present, False otherwise
    """
    required_keys = [
        settings.openai_api_key,
        settings.pinecone_api_key, 
        settings.cohere_api_key,
        settings.llama_cloud_api_key
    ]
    
    missing_keys = [key for key in required_keys if not key or key.startswith("your_")]
    
    if missing_keys:
        print(f"Missing or invalid API keys: {len(missing_keys)} keys need to be configured")
        return False
    
    return True


def get_chunk_metadata_template() -> dict:
    """
    Get the template for chunk-level metadata
    Based on PRD specifications section 3.4
    """
    return {
        "page_number": 0
    }


def get_batch_processing_config() -> dict:
    """
    Get configuration for batch processing system
    
    Returns:
        dict: Batch processing configuration
    """
    return {
        "max_parallel_workflows": settings.max_parallel_workflows,
        "workflow_timeout": settings.workflow_timeout,
        "queue_max_size": settings.queue_max_size,
        "progress_update_interval": settings.progress_update_interval,
        "graceful_shutdown_timeout": settings.graceful_shutdown_timeout,
        "enable_retries": settings.enable_retries,
        "max_retries_per_document": settings.max_retries_per_document
    }


def get_monitoring_config() -> dict:
    """
    Get configuration for progress monitoring system
    
    Returns:
        dict: Monitoring configuration
    """
    return {
        "monitoring_enabled": settings.monitoring_enabled,
        "snapshot_interval": settings.snapshot_interval,
        "progress_history_size": settings.progress_history_size,
        "export_progress_reports": settings.export_progress_reports
    }


def get_resource_limits() -> dict:
    """
    Get system resource limits configuration
    
    Returns:
        dict: Resource limits
    """
    return {
        "memory_limit_mb": settings.memory_limit_mb,
        "disk_space_limit_mb": settings.disk_space_limit_mb,
        "max_file_size_mb": settings.max_file_size_mb,
        "temp_dir": settings.temp_dir
    }


def get_performance_config() -> dict:
    """
    Get performance optimization configuration
    
    Returns:
        dict: Performance settings
    """
    return {
        "use_async_processing": settings.use_async_processing,
        "enable_caching": settings.enable_caching,
        "cache_ttl_seconds": settings.cache_ttl_seconds,
        "optimize_memory": settings.optimize_memory
    }


def get_error_handling_config() -> dict:
    """
    Get error handling configuration
    
    Returns:
        dict: Error handling settings
    """
    return {
        "error_retry_delay": settings.error_retry_delay,
        "max_consecutive_errors": settings.max_consecutive_errors,
        "error_notification_enabled": settings.error_notification_enabled,
        "detailed_error_logging": settings.detailed_error_logging
    }


def validate_parallel_processing_config() -> dict:
    """
    Validate parallel processing configuration and return status
    
    Returns:
        dict: Validation results with status and recommendations
    """
    issues = []
    warnings = []
    
    # Check parallel workflows limit
    if settings.max_parallel_workflows > 4:
        warnings.append("More than 4 parallel workflows may cause resource contention")
    elif settings.max_parallel_workflows < 1:
        issues.append("max_parallel_workflows must be at least 1")
    
    # Check timeout settings
    if settings.workflow_timeout < 300:  # 5 minutes
        warnings.append("Workflow timeout less than 5 minutes may cause premature failures")
    elif settings.workflow_timeout > 3600:  # 1 hour
        warnings.append("Workflow timeout over 1 hour may cause resource waste")
    
    # Check memory limits (if psutil is available)
    if PSUTIL_AVAILABLE:
        available_memory = psutil.virtual_memory().available // (1024 * 1024)
        if settings.memory_limit_mb > available_memory * 0.8:
            warnings.append(f"Memory limit ({settings.memory_limit_mb}MB) exceeds 80% of available memory ({available_memory}MB)")
    else:
        warnings.append("Install 'psutil' package for memory validation")
    
    # Check disk space
    if os.path.exists(settings.temp_dir):
        try:
            _, _, free_space = shutil.disk_usage(settings.temp_dir)
            free_space_mb = free_space // (1024 * 1024)
            if settings.disk_space_limit_mb > free_space_mb * 0.8:
                warnings.append(f"Disk space limit ({settings.disk_space_limit_mb}MB) exceeds 80% of available space ({free_space_mb}MB)")
        except Exception:
            warnings.append(f"Could not check disk space for {settings.temp_dir}")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "recommendations": [
            "Consider adjusting parallel workflows based on available CPU cores",
            "Monitor memory usage during processing",
            "Ensure sufficient disk space for temporary files"
        ]
    }


def create_example_env_file(file_path: str = ".env.example") -> None:
    """
    Create an example environment file with all configuration options
    
    Args:
        file_path: Path to create the example file
    """
    env_content = """# ArXiv RAG System Configuration

# API Keys (required)
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key_here

# Pinecone Configuration
PINECONE_ENVIRONMENT=us-west-2
PINECONE_INDEX_NAME=arxiv-rag
PINECONE_CLOUD=aws
PINECONE_REGION=us-west-2

# Model Configuration
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
COHERE_RERANK_MODEL=rerank-v3.5

# Processing Configuration
MAX_CHUNK_SIZE=1024
CHUNK_OVERLAP=0.15
SEARCH_TOP_K=50
RERANK_TOP_N=10
ANSWER_TEMPERATURE=0.1

# Application Configuration
DEBUG=false
LOG_LEVEL=INFO

# LlamaExtract Configuration
LLAMAEXTRACT_DELAY_SECONDS=15.0

# Parallel Processing Configuration
MAX_PARALLEL_WORKFLOWS=2
WORKFLOW_TIMEOUT=1800.0
QUEUE_MAX_SIZE=20
PROGRESS_UPDATE_INTERVAL=5.0
GRACEFUL_SHUTDOWN_TIMEOUT=300.0
ENABLE_RETRIES=true
MAX_RETRIES_PER_DOCUMENT=2

# Progress Monitoring Configuration
MONITORING_ENABLED=true
SNAPSHOT_INTERVAL=10.0
PROGRESS_HISTORY_SIZE=1000
EXPORT_PROGRESS_REPORTS=true

# Resource Management Configuration
MEMORY_LIMIT_MB=4096
DISK_SPACE_LIMIT_MB=10240
MAX_FILE_SIZE_MB=50.0
TEMP_DIR=/tmp/arxiv_rag

# Performance Optimization Configuration
USE_ASYNC_PROCESSING=true
ENABLE_CACHING=true
CACHE_TTL_SECONDS=3600
OPTIMIZE_MEMORY=true

# Error Handling Configuration
ERROR_RETRY_DELAY=5.0
MAX_CONSECUTIVE_ERRORS=5
ERROR_NOTIFICATION_ENABLED=false
DETAILED_ERROR_LOGGING=true
"""
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(env_content)
    
    print(f"Example environment file created: {file_path}")


class ConfigValidator:
    """Utility class for configuration validation"""
    
    @staticmethod
    def get_missing_api_keys() -> list:
        """Get list of missing API keys"""
        missing = []
        
        if not settings.openai_api_key or settings.openai_api_key.startswith("your_"):
            missing.append("OPENAI_API_KEY")
        if not settings.pinecone_api_key or settings.pinecone_api_key.startswith("your_"):
            missing.append("PINECONE_API_KEY")
        if not settings.cohere_api_key or settings.cohere_api_key.startswith("your_"):
            missing.append("COHERE_API_KEY")
        if not settings.llama_cloud_api_key or settings.llama_cloud_api_key.startswith("your_"):
            missing.append("LLAMA_CLOUD_API_KEY")
            
        return missing
    
    @staticmethod
    def validate_api_keys() -> dict:
        """Validate API keys and return detailed status"""
        keys_status = {}
        
        keys_to_check = {
            "OPENAI_API_KEY": settings.openai_api_key,
            "PINECONE_API_KEY": settings.pinecone_api_key,
            "COHERE_API_KEY": settings.cohere_api_key,
            "LLAMA_CLOUD_API_KEY": settings.llama_cloud_api_key
        }
        
        for key_name, key_value in keys_to_check.items():
            is_valid = bool(key_value and not key_value.startswith("your_"))
            keys_status[key_name] = is_valid
            
        return keys_status
    
    @staticmethod
    def validate_configuration() -> dict:
        """Comprehensive configuration validation"""
        return {
            "api_keys": ConfigValidator.validate_api_keys(),
            "parallel_processing": validate_parallel_processing_config(),
            "system_requirements": {
                "python_version": ">=3.8",
                "memory_recommended": "8GB+",
                "disk_space_recommended": "20GB+"
            }
        }