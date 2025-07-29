"""
Configuration module for the ArXiv RAG System

This module handles all configuration settings for the application,
including API keys, model parameters, and system settings.
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

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
    - Always respond in Russian language
    - Exclude abstract and evaluative judgments
    - Provide only concrete facts, numbers, methods, results
    - Specify sources (pages, tables, sections)
    
 RESPONSE STRUCTURE:
    
    **Direct answer to the question:**
     [Provide a specific answer to the question with factual data, composing a complete response of 3-5 sentences. The response should not contain evaluative or abstract expressions. The answer should contain a logical narrative thread. You can also use knowledge from the mainFindings metadata if necessary.]
    
    **Additional details:**
    [Relevant technical information from the text fragment, e.g., formulas, tables with variable descriptions, or information from the results metadata]
    
    **Research context:**
    [Synthesis of information from the Abstract and methodology metadata.]
    
    **Sources:**
    [1-2 Specific references to pages and sections of documents, indicating authors and titles of works + references from metadata for further study]

    
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
    
    REMEMBER: Your goal is to be a precise information search tool that helps users quickly get specific answers to their questions from scientific materials. Respond exclusively in Russian Language.
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
