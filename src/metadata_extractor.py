"""
Metadata extractor module using LlamaExtract for structured metadata extraction

This module provides metadata extraction capabilities using LlamaExtract to extract
structured metadata from documents according to a defined JSON schema.
Based on PRD specifications with citations, reasoning, and full metadata schema.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, ValidationError
from llama_index.core import Document

from .config import settings, get_metadata_schema



class MetadataExtractorConfig(BaseModel):
    """Configuration for MetadataExtractor"""
    
    api_key: str = Field(default_factory=lambda: settings.llama_cloud_api_key)
    extraction_mode: str = Field(default=settings.llamaextract_mode)  # "BALANCED"
    extraction_target: str = Field(default=settings.llamaextract_target)  # "PER_DOC"
    use_reasoning: bool = Field(default=settings.llamaextract_use_reasoning)  # True
    cite_sources: bool = Field(default=settings.llamaextract_cite_sources)  # True
    delay_seconds: float = Field(default=settings.llamaextract_delay_seconds)  # 15.0
    schema: Dict[str, Any] = Field(default_factory=get_metadata_schema)


class ExtractedMetadata(BaseModel):
    """
    Pydantic model for extracted metadata based on PRD schema
    This mirrors the JSON schema from config.py for validation
    """
    
    title: str
    authors: List[Dict[str, Optional[str]]]
    abstract: str
    mainFindings: List[str]
    keywords: Optional[List[str]] = None
    methodology: Optional[Dict[str, Any]] = None
    results: Optional[List[Dict[str, Optional[str]]]] = None
    discussion: Optional[Dict[str, List[str]]] = None
    references: Optional[List[Dict[str, Optional[str]]]] = None
    publication: Optional[Dict[str, Optional[str]]] = None
    
    # Additional fields for reasoning and citations (PRD requirement)
    _reasoning: Optional[Dict[str, str]] = None
    _citations: Optional[Dict[str, List[Dict[str, Any]]]] = None
    
    class Config:
        extra = "allow"  # Allow additional fields from reasoning and citations


class MetadataExtractor:
    """
    Metadata extractor using LlamaExtract
    
    This class handles structured metadata extraction from documents:
    - Extraction by detailed JSON schema (PRD section 3.1)
    - BALANCED mode for optimal accuracy/performance balance
    - PER_DOC target for document-level metadata
    - Citations and reasoning enabled for scientific reliability
    
    Based on PRD specifications for ArXiv scientific document analysis.
    """
    
    def __init__(self, config: Optional[MetadataExtractorConfig] = None):
        """
        Initialize MetadataExtractor with LlamaExtract integration
        
        Args:
            config: Optional configuration override
        """
        self.config = config or MetadataExtractorConfig()
        
        # Validate API key
        if not self.config.api_key or self.config.api_key.startswith("your_"):
            raise ValueError(
                "LLAMA_CLOUD_API_KEY is required. "
                "Set it in environment variables or .env file"
            )
        
        # Initialize LlamaExtract
        self._initialize_llama_extract()
        
    
    def _initialize_llama_extract(self):
        """
        Initialize LlamaExtract with configuration
        """
        try:
            # Import LlamaExtract - handle potential import issues gracefully
            try:
                from llama_cloud_services import LlamaExtract
                from llama_cloud_services.extract.extract import SourceText
            except ImportError:
                # Fallback implementation using LlamaIndex extractors
                self._use_fallback_implementation()
                return
            
            # Initialize LlamaExtract (new 2025 API)
            self.extractor = LlamaExtract()
            
            # Create Pydantic schema from JSON schema
            pydantic_schema = self._create_pydantic_schema()
            
            # Get existing agent or create new one
            self.agent = self._get_or_create_agent(
                agent_name="arxiv_metadata_extractor",
                pydantic_schema=pydantic_schema
            )
            
            self.use_fallback = False
            
        except Exception as e:
            self._use_fallback_implementation()
    
    def _create_pydantic_schema(self):
        """
        Create Pydantic schema from JSON schema for LlamaExtract 2025 API
        
        LlamaExtract requires Pydantic BaseModel instead of JSON schema.
        Converts the JSON schema defined in config to Pydantic format.
        
        Returns:
            Pydantic BaseModel class for document metadata
        """
        from pydantic import BaseModel, Field
        from typing import List, Optional
        
        class Author(BaseModel):
            name: str = Field(description="Full name of the author")
            affiliation: Optional[str] = Field(default=None, description="Institution or organization")
            email: Optional[str] = Field(default=None, description="Contact email")
        
        class Methodology(BaseModel):
            approach: Optional[str] = Field(default=None, description="Overall research approach")
            participants: Optional[str] = Field(default=None, description="Study participants or data sources")
            methods: Optional[List[str]] = Field(default=None, description="Specific methods and techniques")
        
        class Result(BaseModel):
            finding: str = Field(description="Description of the specific result")
            significance: Optional[str] = Field(default=None, description="Statistical significance")
            supportingData: Optional[str] = Field(default=None, description="Relevant statistics or data")
        
        class Publication(BaseModel):
            journal: Optional[str] = Field(default=None, description="Name of journal or conference")
            year: str = Field(description="Year of publication")
            doi: Optional[str] = Field(default=None, description="Digital Object Identifier")
            url: Optional[str] = Field(default=None, description="URL where paper can be accessed")
        
        class DocumentMetadata(BaseModel):
            title: str = Field(description="The full title of the research paper")
            authors: List[Author] = Field(description="List of all authors of the paper")
            abstract: str = Field(description="Complete abstract or summary of the paper")
            mainFindings: List[str] = Field(description="Key findings, conclusions, or contributions")
            keywords: Optional[List[str]] = Field(default=None, description="Key terms and phrases")
            methodology: Optional[Methodology] = Field(default=None, description="Research methods used")
            results: Optional[List[Result]] = Field(default=None, description="Main results and outcomes")
            publication: Optional[Publication] = Field(default=None, description="Publication information")
        
        return DocumentMetadata
    
    def _get_or_create_agent(self, agent_name: str, pydantic_schema):
        """
        Get existing agent or create a new one if it doesn't exist
        
        Args:
            agent_name: Name of the agent
            pydantic_schema: Pydantic schema for the agent
            
        Returns:
            Agent instance
            
        Raises:
            Exception: If unable to get or create agent
        """
        try:
            # First, try to get existing agent
            
            # Try to list existing agents and find the one with our name
            try:
                # Note: This assumes the LlamaExtract API has a method to list agents
                # If not available, we'll catch the exception and proceed with creation
                existing_agents = self.extractor.list_agents()
                for agent in existing_agents:
                    if agent.name == agent_name:
                        return agent
                
            except AttributeError:
                # list_agents method might not exist, proceed with creation attempt
                pass
            except Exception as e:
                pass  # Handle agent listing errors gracefully  # Handle list_agents API errors gracefully
            
            # If no existing agent found, try to create a new one
            agent = self.extractor.create_agent(
                name=agent_name,
                data_schema=pydantic_schema
            )
            return agent
            
        except Exception as e:
            # Check if it's a 409 Conflict error (agent already exists)
            if "409" in str(e) or "already exists" in str(e).lower():
                
                # Try to get the existing agent by name
                try:
                    # If there's a get_agent method
                    if hasattr(self.extractor, 'get_agent'):
                        agent = self.extractor.get_agent(agent_name)
                        return agent
                    else:
                        # If no direct get method, try to use the agent anyway
                        # This is a workaround - we'll create a mock agent reference
                        
                        # Create a minimal agent-like object that can be used for extraction
                        class ExistingAgent:
                            def __init__(self, extractor, name):
                                self.extractor = extractor
                                self.name = name
                            
                            def extract(self, source_text):
                                # Use the extractor directly with the known agent name
                                return self.extractor.extract(source_text, agent_name=self.name)
                        
                        agent = ExistingAgent(self.extractor, agent_name)
                        return agent
                        
                except Exception as get_error:
                    raise Exception(f"Agent exists but cannot be retrieved: {str(get_error)}")
            else:
                # Some other error occurred
                raise Exception(f"Failed to create or get agent: {str(e)}")
    
    def _use_fallback_implementation(self):
        """
        Use fallback implementation with LlamaIndex extractors
        """
        try:
            from llama_index.core.extractors import (
                TitleExtractor,
                QuestionsAnsweredExtractor,
                SummaryExtractor,
                KeywordExtractor
            )
            from llama_index.llms.openai import OpenAI
            
            # Initialize LLM for extractors
            llm = OpenAI(
                api_key=settings.openai_api_key,
                model=settings.openai_chat_model,
                temperature=settings.answer_temperature
            )
            
            # Initialize extractors
            self.title_extractor = TitleExtractor(nodes=5, llm=llm)
            self.qa_extractor = QuestionsAnsweredExtractor(questions=3, llm=llm)
            self.summary_extractor = SummaryExtractor(summaries=["self"], llm=llm)
            self.keyword_extractor = KeywordExtractor(keywords=10, llm=llm)
            
            self.use_fallback = True
            
        except Exception as e:
            raise
    
    def extract_metadata(self, document: Union[Document, str]) -> ExtractedMetadata:
        """
        Extract metadata from document using LlamaExtract or fallback
        
        Args:
            document: Document object or text string
            
        Returns:
            ExtractedMetadata object with extracted information
            
        Raises:
            Exception: If metadata extraction fails
        """
        try:
            
            if self.use_fallback:
                return self._extract_with_fallback(document)
            else:
                return self._extract_with_llama_extract(document)
                
        except Exception as e:
            raise Exception(f"Error extracting metadata: {str(e)}")
    
    def _extract_with_llama_extract(self, document: Union[Document, str]) -> ExtractedMetadata:
        """
        Extract metadata using LlamaExtract
        
        Args:
            document: Document object or text string
            
        Returns:
            ExtractedMetadata object with extracted information
        """
        try:
            from llama_cloud_services.extract.extract import SourceText
            
            # Convert to string if Document object
            if isinstance(document, Document):
                doc_text = document.text
                doc_metadata = getattr(document, 'metadata', {})
            else:
                doc_text = document
                doc_metadata = {}
            
            # Add delay before LlamaExtract API call to reduce polling frequency
            time.sleep(self.config.delay_seconds)
            
            # Perform extraction with new LlamaExtract 2025 API using SourceText
            result = self.agent.extract(SourceText(text_content=doc_text))
            
            # Extract data from new API result format
            if hasattr(result, 'data'):
                extracted_data = result.data
            elif isinstance(result, dict):
                extracted_data = result
            else:
                extracted_data = result.__dict__ if hasattr(result, '__dict__') else {}
            
            # Process result to include reasoning and citations
            processed_result = self._process_llama_extract_result(extracted_data)
            
            # Create and validate ExtractedMetadata
            metadata = ExtractedMetadata(**processed_result)
            
            return metadata
            
        except Exception as e:
            raise
    
    def _extract_with_fallback(self, document: Union[Document, str]) -> ExtractedMetadata:
        """
        Extract metadata using fallback LlamaIndex extractors
        
        Args:
            document: Document object or text string
            
        Returns:
            ExtractedMetadata object with extracted information
        """
        try:
            from llama_index.core.node_parser import SentenceSplitter
            
            # Convert to Document if string
            if isinstance(document, str):
                doc = Document(text=document)
            else:
                doc = document
            
            # Split into nodes for processing
            splitter = SentenceSplitter(chunk_size=2048)
            nodes = splitter.get_nodes_from_documents([doc])
            
            # Extract metadata using different extractors
            metadata_dict = {}
            
            # Extract title
            try:
                title_nodes = self.title_extractor.extract(nodes[:1])
                metadata_dict["title"] = title_nodes[0].get("document_title", "Unknown Title")
            except Exception:
                metadata_dict["title"] = "Unknown Title"
            
            # Extract keywords
            try:
                keyword_nodes = self.keyword_extractor.extract(nodes[:1])
                keywords = keyword_nodes[0].get("excerpt_keywords", "").split(", ")
                metadata_dict["keywords"] = [k.strip() for k in keywords if k.strip()]
            except Exception:
                metadata_dict["keywords"] = []
            
            # Extract questions/findings
            try:
                qa_nodes = self.qa_extractor.extract(nodes[:1])
                questions = qa_nodes[0].get("questions_this_excerpt_can_answer", "")
                metadata_dict["mainFindings"] = [questions] if questions else []
            except Exception:
                metadata_dict["mainFindings"] = []
            
            # Extract summary as abstract
            try:
                summary_nodes = self.summary_extractor.extract(nodes[:1])
                metadata_dict["abstract"] = summary_nodes[0].get("section_summary", "")
            except Exception:
                metadata_dict["abstract"] = ""
            
            # Set default values for required fields
            metadata_dict.setdefault("authors", [{"name": "Unknown Author"}])
            metadata_dict.setdefault("title", "Unknown Title")
            metadata_dict.setdefault("abstract", "")
            metadata_dict.setdefault("mainFindings", [])
            
            # Create and validate ExtractedMetadata
            metadata = ExtractedMetadata(**metadata_dict)
            
            return metadata
            
        except Exception as e:
            raise
    
    def _process_llama_extract_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process LlamaExtract result to include reasoning and citations
        
        Args:
            result: Raw result from LlamaExtract
            
        Returns:
            Processed result dictionary
        """
        processed = result.copy()
        
        # Extract reasoning fields (fields ending with "_reasoning")
        reasoning = {}
        citations = {}
        
        keys_to_remove = []
        for key, value in result.items():
            if key.endswith("_reasoning"):
                base_key = key.replace("_reasoning", "")
                reasoning[base_key] = value
                keys_to_remove.append(key)
            elif key.endswith("_citations"):
                base_key = key.replace("_citations", "")
                citations[base_key] = value
                keys_to_remove.append(key)
        
        # Remove reasoning and citation fields from main result
        for key in keys_to_remove:
            processed.pop(key, None)
        
        # Add reasoning and citations as private fields
        if reasoning:
            processed["_reasoning"] = reasoning
        if citations:
            processed["_citations"] = citations
        
        # Clean None values from nested structures to ensure compatibility
        processed = self._clean_none_values(processed)
        
        return processed
    
    def _clean_none_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean None values from nested dictionaries to ensure Pydantic validation passes
        
        Args:
            data: Dictionary to clean
            
        Returns:
            Cleaned dictionary
        """
        cleaned = {}
        
        for key, value in data.items():
            if value is None:
                # Keep None values for optional fields
                cleaned[key] = value
            elif isinstance(value, dict):
                # Recursively clean nested dictionaries
                cleaned_dict = {}
                for nested_key, nested_value in value.items():
                    if nested_value is not None:
                        cleaned_dict[nested_key] = nested_value
                cleaned[key] = cleaned_dict if cleaned_dict else None
            elif isinstance(value, list) and value:
                # Clean lists of dictionaries (like results, references)
                cleaned_list = []
                for item in value:
                    if isinstance(item, dict):
                        cleaned_item = {}
                        for item_key, item_value in item.items():
                            if item_value is not None:
                                cleaned_item[item_key] = item_value
                        if cleaned_item:  # Only add non-empty dictionaries
                            cleaned_list.append(cleaned_item)
                    else:
                        cleaned_list.append(item)
                cleaned[key] = cleaned_list if cleaned_list else None
            else:
                # Keep other values as is
                cleaned[key] = value
        
        return cleaned
    
    def extract_batch(self, documents: List[Union[Document, str]]) -> List[ExtractedMetadata]:
        """
        Extract metadata from multiple documents in batch
        
        Args:
            documents: List of Document objects or text strings
            
        Returns:
            List of ExtractedMetadata objects
        """
        results = []
        
        for i, doc in enumerate(documents):
            try:
                metadata = self.extract_metadata(doc)
                results.append(metadata)
            except Exception as e:
                # Create empty metadata for failed extraction
                empty_metadata = ExtractedMetadata(
                    title="Extraction Failed",
                    authors=[{"name": "Unknown"}],
                    abstract="Failed to extract abstract",
                    mainFindings=["Extraction failed"]
                )
                results.append(empty_metadata)
        
        return results
    
    def validate_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Validate extracted metadata against schema
        
        Args:
            metadata: Metadata dictionary to validate
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            ExtractedMetadata(**metadata)
            return True
        except ValidationError as e:
            return False
    
    def get_extraction_stats(self, metadata_list: List[ExtractedMetadata]) -> Dict[str, Any]:
        """
        Get statistics about extracted metadata
        
        Args:
            metadata_list: List of extracted metadata objects
            
        Returns:
            Dictionary with extraction statistics
        """
        if not metadata_list:
            return {"total_documents": 0}
        
        total_docs = len(metadata_list)
        total_authors = sum(len(meta.authors) for meta in metadata_list)
        total_findings = sum(len(meta.mainFindings) for meta in metadata_list)
        total_keywords = sum(len(meta.keywords or []) for meta in metadata_list)
        
        # Check for reasoning and citations availability
        with_reasoning = sum(1 for meta in metadata_list if hasattr(meta, '_reasoning') and meta._reasoning)
        with_citations = sum(1 for meta in metadata_list if hasattr(meta, '_citations') and meta._citations)
        
        stats = {
            "total_documents": total_docs,
            "total_authors": total_authors,
            "total_findings": total_findings,
            "total_keywords": total_keywords,
            "average_authors_per_doc": total_authors / total_docs,
            "average_findings_per_doc": total_findings / total_docs,
            "average_keywords_per_doc": total_keywords / total_docs,
            "documents_with_reasoning": with_reasoning,
            "documents_with_citations": with_citations,
            "extraction_mode": self.config.extraction_mode,
            "use_fallback": self.use_fallback
        }
        
        return stats




# Convenience functions for easy usage
def extract_metadata(document: Union[Document, str], **kwargs) -> ExtractedMetadata:
    """
    Convenience function to extract metadata with default settings
    
    Args:
        document: Document object or text string
        **kwargs: Additional configuration options
        
    Returns:
        ExtractedMetadata object
    """
    config = MetadataExtractorConfig(**kwargs)
    extractor = MetadataExtractor(config)
    return extractor.extract_metadata(document)


