"""
Vector store module using Pinecone and OpenAI embeddings

This module provides vector database functionality using Pinecone with OpenAI
text-embedding-3-large for document indexing and similarity search.
Based on PRD specifications with 1536 dimensions and cosine similarity.
"""

import logging
import time
import uuid
from typing import List, Dict, Any, Optional, Union, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pydantic import BaseModel, Field
from llama_index.core.schema import TextNode
from llama_index.core import Document

from .config import settings

# Set up logger
logger = logging.getLogger(__name__)


class VectorStoreConfig(BaseModel):
    """Configuration for VectorStore"""
    
    # Pinecone configuration
    api_key: str = Field(default_factory=lambda: settings.pinecone_api_key)
    environment: str = Field(default=settings.pinecone_environment)
    index_name: str = Field(default=settings.pinecone_index_name)
    cloud: str = Field(default=settings.pinecone_cloud)
    region: str = Field(default=settings.pinecone_region)
    
    # Vector configuration
    dimension: int = Field(default=settings.vector_dimension)  # 1536 for text-embedding-3-large
    metric: str = Field(default=settings.similarity_metric)  # cosine
    
    # OpenAI configuration
    openai_api_key: str = Field(default_factory=lambda: settings.openai_api_key)
    embedding_model: str = Field(default=settings.openai_embedding_model)
    
    # Rate limiting configuration
    batch_size: int = Field(default=settings.batch_size)  # 100
    batch_delay: float = Field(default=settings.batch_delay)  # 1.0 seconds
    max_retries: int = Field(default=3)
    retry_delay: float = Field(default=2.0)
    
    # Search configuration
    search_top_k: int = Field(default=settings.search_top_k)  # 50


class EmbeddingGenerator:
    """
    Generate embeddings using OpenAI text-embedding-3-large
    """
    
    def __init__(self, config: VectorStoreConfig):
        """
        Initialize embedding generator
        
        Args:
            config: Vector store configuration
        """
        self.config = config
        
        # Validate API key
        if not self.config.openai_api_key or self.config.openai_api_key.startswith("your_"):
            raise ValueError(
                "OPENAI_API_KEY is required. "
                "Set it in environment variables or .env file"
            )
        
        # Initialize OpenAI client
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.config.openai_api_key)
        except ImportError:
            raise ImportError("OpenAI library not found. Install with: pip install openai")
        
        logger.info(f"EmbeddingGenerator initialized with model: {self.config.embedding_model}")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for single text using OpenAI
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            Exception: If embedding generation fails
        """
        try:
            # Clean and validate text
            text = text.strip()
            if not text:
                raise ValueError("Empty text provided for embedding")
            
            # Truncate text if too long (OpenAI limit is ~8191 tokens)
            if len(text) > 30000:  # Rough character limit
                text = text[:30000] + "..."
                logger.warning("Text truncated for embedding generation")
            
            response = self.client.embeddings.create(
                input=text,
                model=self.config.embedding_model
            )
            
            embedding = response.data[0].embedding
            
            # Validate embedding dimension
            if len(embedding) != self.config.dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: got {len(embedding)}, "
                    f"expected {self.config.dimension}"
                )
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise Exception(f"Error generating embedding: {str(e)}")
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts in batch
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            Exception: If batch embedding generation fails
        """
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            
            # Clean and validate texts
            cleaned_texts = []
            for text in texts:
                text = text.strip()
                if text:
                    # Truncate if too long
                    if len(text) > 30000:
                        text = text[:30000] + "..."
                    cleaned_texts.append(text)
                else:
                    cleaned_texts.append("Empty text")  # Placeholder for empty texts
            
            if not cleaned_texts:
                raise ValueError("No valid texts provided for embedding")
            
            # Process in batches to respect rate limits
            all_embeddings = []
            
            for i in range(0, len(cleaned_texts), self.config.batch_size):
                batch = cleaned_texts[i:i + self.config.batch_size]
                
                # Generate embeddings for batch
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.config.embedding_model
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Add delay between batches
                if i + self.config.batch_size < len(cleaned_texts):
                    time.sleep(self.config.batch_delay)
                    logger.info(f"Processed batch {i//self.config.batch_size + 1}")
            
            # Validate all embeddings
            for i, embedding in enumerate(all_embeddings):
                if len(embedding) != self.config.dimension:
                    raise ValueError(
                        f"Embedding {i} dimension mismatch: got {len(embedding)}, "
                        f"expected {self.config.dimension}"
                    )
            
            logger.info(f"Generated {len(all_embeddings)} embeddings successfully")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise Exception(f"Error generating batch embeddings: {str(e)}")


class PineconeManager:
    """
    Manage Pinecone vector database operations
    """
    
    def __init__(self, config: VectorStoreConfig):
        """
        Initialize Pinecone manager
        
        Args:
            config: Vector store configuration
        """
        self.config = config
        
        # Validate API key
        if not self.config.api_key or self.config.api_key.startswith("your_"):
            raise ValueError(
                "PINECONE_API_KEY is required. "
                "Set it in environment variables or .env file"
            )
        
        # Initialize Pinecone
        try:
            from pinecone import Pinecone, ServerlessSpec
            self.Pinecone = Pinecone
            self.ServerlessSpec = ServerlessSpec
            
            self.pc = Pinecone(api_key=self.config.api_key)
        except ImportError:
            raise ImportError("Pinecone library not found. Install with: pip install pinecone")
        
        # Create or connect to index
        self._setup_index()
        
        logger.info(f"PineconeManager initialized with index: {self.config.index_name}")
    
    def _setup_index(self):
        """Set up Pinecone index"""
        try:
            # List existing indexes
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.config.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.config.index_name}")
                
                # Create new index
                self.pc.create_index(
                    name=self.config.index_name,
                    dimension=self.config.dimension,
                    metric=self.config.metric,
                    spec=self.ServerlessSpec(
                        cloud=self.config.cloud,
                        region=self.config.region
                    )
                )
                
                # Wait for index to be ready
                logger.info("Waiting for index to be ready...")
                time.sleep(10)
                
                # Wait until index is ready
                while not self.pc.describe_index(self.config.index_name).status['ready']:
                    time.sleep(1)
                
                logger.info("Index created successfully")
            else:
                logger.info(f"Using existing index: {self.config.index_name}")
            
            # Connect to index
            self.index = self.pc.Index(self.config.index_name)
            
            # Get index stats
            stats = self.index.describe_index_stats()
            logger.info(f"Index stats: {stats}")
            
        except Exception as e:
            logger.error(f"Error setting up Pinecone index: {str(e)}")
            raise
    
    def upsert_vectors(self, vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Upsert vectors to Pinecone
        
        Args:
            vectors: List of vector dictionaries with id, values, metadata
            
        Returns:
            Upsert response
            
        Raises:
            Exception: If upsert fails
        """
        try:
            logger.info(f"Starting upsert process for {len(vectors)} vectors to Pinecone")
            
            if not vectors:
                return {"upserted_count": 0}
            
            # Check for duplicate IDs in the batch
            vector_ids = [v["id"] for v in vectors]
            unique_ids = set(vector_ids)
            if len(unique_ids) != len(vector_ids):
                duplicates = [id for id in vector_ids if vector_ids.count(id) > 1]
                logger.warning(f"DUPLICATE IDs in batch: {set(duplicates)}")
            
            logger.info("=== Pinecone Upsert Details ===")
            
            # Process in batches
            total_upserted = 0
            batch_number = 0
            
            for i in range(0, len(vectors), self.config.batch_size):
                batch = vectors[i:i + self.config.batch_size]
                batch_number += 1
                
                # Log batch details
                batch_ids = [v["id"] for v in batch]
                logger.info(f"Preparing batch {batch_number}: {len(batch)} vectors")
                logger.info(f"Batch {batch_number} IDs: {batch_ids}")
                
                # Log sample vector details from batch
                if batch:
                    sample_vector = batch[0]
                    metadata_keys = list(sample_vector.get("metadata", {}).keys())
                    embedding_size = len(sample_vector.get("values", []))
                    logger.info(
                        f"Sample vector from batch {batch_number}: id={sample_vector['id']}, "
                        f"embedding_dims={embedding_size}, metadata_keys={metadata_keys}"
                    )
                
                # Retry logic for batch upsert
                for attempt in range(self.config.max_retries):
                    try:
                        logger.info(f"Upserting batch {batch_number} to Pinecone (attempt {attempt + 1})")
                        response = self.index.upsert(vectors=batch)
                        
                        upserted_count = response.get('upserted_count', len(batch))
                        total_upserted += upserted_count
                        
                        logger.info(
                            f"Batch {batch_number} success: upserted {upserted_count} vectors, "
                            f"response: {response}"
                        )
                        break
                    except Exception as e:
                        if attempt < self.config.max_retries - 1:
                            logger.warning(f"Batch {batch_number} attempt {attempt + 1} failed: {str(e)}. Retrying...")
                            time.sleep(self.config.retry_delay)
                        else:
                            logger.error(f"Batch {batch_number} failed after {self.config.max_retries} attempts: {str(e)}")
                            raise
                
                # Add delay between batches
                if i + self.config.batch_size < len(vectors):
                    time.sleep(self.config.batch_delay)
                    logger.info(f"Completed batch {batch_number}, waiting {self.config.batch_delay}s before next batch")
            
            logger.info("=== End Pinecone Upsert Details ===")
            logger.info(f"Upsert completed: {total_upserted} vectors processed across {batch_number} batches")
            return {"upserted_count": total_upserted}
            
        except Exception as e:
            logger.error(f"Error upserting vectors: {str(e)}")
            raise Exception(f"Error upserting vectors: {str(e)}")
    
    def query_vectors(self, query_vector: List[float], top_k: int = None, 
                     filter_dict: Optional[Dict[str, Any]] = None, 
                     include_metadata: bool = True) -> Dict[str, Any]:
        """
        Query vectors from Pinecone
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_dict: Metadata filter dictionary
            include_metadata: Whether to include metadata in results
            
        Returns:
            Query results
            
        Raises:
            Exception: If query fails
        """
        try:
            if top_k is None:
                top_k = self.config.search_top_k
            
            # Validate query vector
            if len(query_vector) != self.config.dimension:
                raise ValueError(
                    f"Query vector dimension mismatch: got {len(query_vector)}, "
                    f"expected {self.config.dimension}"
                )
            
            logger.info(f"Querying Pinecone with top_k: {top_k}")
            
            # Perform query
            response = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=include_metadata,
                filter=filter_dict
            )
            
            logger.info(f"Query returned {len(response.matches)} results")
            return response
            
        except Exception as e:
            logger.error(f"Error querying vectors: {str(e)}")
            raise Exception(f"Error querying vectors: {str(e)}")
    
    def delete_vectors(self, ids: List[str]) -> Dict[str, Any]:
        """
        Delete vectors from Pinecone
        
        Args:
            ids: List of vector IDs to delete
            
        Returns:
            Delete response
        """
        try:
            logger.info(f"Deleting {len(ids)} vectors from Pinecone")
            
            response = self.index.delete(ids=ids)
            
            logger.info("Vectors deleted successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error deleting vectors: {str(e)}")
            raise Exception(f"Error deleting vectors: {str(e)}")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get Pinecone index statistics
        
        Returns:
            Index statistics
        """
        try:
            stats = self.index.describe_index_stats()
            return stats
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {}


class VectorStore:
    """
    Main vector store class combining Pinecone and OpenAI embeddings
    
    This class provides a complete vector database solution:
    - Automatic embedding generation with OpenAI text-embedding-3-large
    - Vector storage and retrieval with Pinecone
    - Batch processing with rate limiting
    - Metadata filtering and search
    """
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """
        Initialize VectorStore
        
        Args:
            config: Optional configuration override
        """
        self.config = config or VectorStoreConfig()
        
        # Initialize components
        self.embedding_generator = EmbeddingGenerator(self.config)
        self.pinecone_manager = PineconeManager(self.config)
        
        logger.info("VectorStore initialized successfully")
    
    def _serialize_metadata_for_pinecone(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize metadata to Pinecone-compatible format
        
        Pinecone only supports flat metadata with simple types:
        - string, number, boolean, list of strings
        - No nested objects or complex types allowed
        
        Args:
            metadata: Original metadata dictionary
            
        Returns:
            Serialized metadata compatible with Pinecone
        """
        import json
        
        serialized = {}
        
        for key, value in metadata.items():
            if key == "authors":
                # Convert [{"name": "John"}] to ["John"]
                if isinstance(value, list):
                    author_names = []
                    for author in value:
                        if isinstance(author, dict):
                            author_names.append(author.get("name", "Unknown"))
                        else:
                            author_names.append(str(author))
                    serialized[key] = author_names
                else:
                    serialized[key] = str(value)
            elif key in ["mainFindings", "keywords"] and isinstance(value, list):
                # Keep lists of strings as is
                serialized[key] = [str(item) for item in value]
            elif isinstance(value, dict):
                # Serialize complex objects to JSON string
                serialized[key] = json.dumps(value)
            elif isinstance(value, list):
                # Convert other lists to list of strings
                serialized[key] = [str(item) for item in value]
            elif value is None:
                # Skip null values (not supported by Pinecone)
                continue
            else:
                # Keep simple types as is
                serialized[key] = value
        
        return serialized
    
    def index_documents(self, documents: List[Union[Document, TextNode]]) -> Dict[str, Any]:
        """
        Index documents in the vector store
        
        Args:
            documents: List of Document or TextNode objects
            
        Returns:
            Indexing results
        """
        try:
            logger.info(f"Starting indexing process for {len(documents)} documents")
            
            if not documents:
                return {"indexed_count": 0}
            
            # Prepare vectors for upsert
            vectors = []
            texts = []
            doc_ids_seen = set()  # Track IDs to detect duplicates
            
            logger.info("=== Vector Preparation Details ===")
            
            for i, doc in enumerate(documents):
                # Extract text and metadata
                if isinstance(doc, TextNode):
                    text = doc.text
                    metadata = doc.metadata or {}
                    doc_id = doc.id_ or f"doc_{i}_{uuid.uuid4().hex[:8]}"
                    doc_type = "TextNode"
                else:  # Document
                    text = doc.text
                    metadata = doc.metadata or {}
                    doc_id = f"doc_{i}_{uuid.uuid4().hex[:8]}"
                    doc_type = "Document"
                
                # Check for duplicate IDs
                if doc_id in doc_ids_seen:
                    logger.warning(f"DUPLICATE ID DETECTED: {doc_id} - this may cause vector overwrites!")
                else:
                    doc_ids_seen.add(doc_id)
                
                # Log document processing details
                page_number = metadata.get("page_number", 0)
                text_preview = text[:100].replace('\n', ' ').replace('\r', '') + "..." if len(text) > 100 else text
                
                logger.info(
                    f"Processing doc {i}: type={doc_type}, id={doc_id}, "
                    f"page={page_number}, "
                    f"size={len(text)} chars, preview=\"{text_preview}\""
                )
                
                texts.append(text)
                
                # Serialize metadata for Pinecone compatibility
                serialized_metadata = self._serialize_metadata_for_pinecone(metadata)
                
                # Prepare vector data
                vector_data = {
                    "id": doc_id,
                    "metadata": {
                        **serialized_metadata,
                        "text": text[:1000],  # Store first 1000 chars for reference
                        "text_length": len(text),
                        "doc_index": i
                    }
                }
                vectors.append(vector_data)
                
                # Log vector metadata details
                metadata_keys = list(serialized_metadata.keys())
                logger.info(
                    f"Created vector data for {doc_id}: metadata_keys={metadata_keys}, "
                    f"text_stored={len(text[:1000])} chars"
                )
            
            logger.info(f"=== End Vector Preparation ===")
            logger.info(f"Prepared {len(vectors)} vectors, detected {len(doc_ids_seen)} unique IDs")
            
            # Generate embeddings in batch
            logger.info(f"Generating embeddings for {len(texts)} texts...")
            embeddings = self.embedding_generator.get_embeddings_batch(texts)
            logger.info(f"Generated {len(embeddings)} embeddings successfully")
            
            # Add embeddings to vector data
            logger.info("Adding embeddings to vector data...")
            for i, (vector_data, embedding) in enumerate(zip(vectors, embeddings)):
                vector_data["values"] = embedding
                logger.info(f"Vector {i} ({vector_data['id']}): embedding_dims={len(embedding)}")
            
            logger.info(f"All {len(vectors)} vectors prepared with embeddings")
            
            # Upsert to Pinecone
            result = self.pinecone_manager.upsert_vectors(vectors)
            
            logger.info(f"Successfully indexed {result.get('upserted_count', 0)} documents")
            return {
                "indexed_count": result.get('upserted_count', 0),
                "total_documents": len(documents),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            return {
                "indexed_count": 0,
                "total_documents": len(documents),
                "success": False,
                "error": str(e)
            }
    
    def index_chunks(self, chunks: List[TextNode]) -> Dict[str, Any]:
        """
        Index chunks (TextNode objects) in the vector store
        
        Args:
            chunks: List of TextNode objects from chunking
            
        Returns:
            Indexing results
        """
        return self.index_documents(chunks)
    
    def search(self, query: str, top_k: int = None, 
              filter_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search for similar documents
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filter_dict: Metadata filter dictionary
            
        Returns:
            Search results with scores and metadata
        """
        try:
            if top_k is None:
                top_k = self.config.search_top_k
            
            logger.info(f"=== VECTOR SEARCH START ===")
            logger.info(f"Query: {query[:100]}...")
            logger.info(f"Query length: {len(query)} chars, top_k: {top_k}")
            if filter_dict:
                logger.info(f"Filters applied: {filter_dict}")
            
            # Generate query embedding
            query_embedding = self.embedding_generator.get_embedding(query)
            
            # Search in Pinecone
            results = self.pinecone_manager.query_vectors(
                query_vector=query_embedding,
                top_k=top_k,
                filter_dict=filter_dict,
                include_metadata=True
            )
            
            # Detailed logging of search results
            logger.info(f"Raw Pinecone results: {len(results.matches)} matches")
            
            # Analyze for duplicates
            result_ids = [match.id for match in results.matches]
            unique_ids = set(result_ids)
            if len(unique_ids) != len(result_ids):
                duplicates = [id for id in result_ids if result_ids.count(id) > 1]
                logger.warning(f"DUPLICATES FOUND in vector search: {set(duplicates)}")
                logger.warning(f"Total results: {len(result_ids)}, Unique IDs: {len(unique_ids)}")
            else:
                logger.info(f"No duplicates in vector search - all {len(result_ids)} results have unique IDs")
            
            # Format results with detailed logging
            formatted_results = []
            for i, match in enumerate(results.matches):
                formatted_result = {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata,
                    "text": match.metadata.get("text", "")
                }
                formatted_results.append(formatted_result)
                
                # Log each result
                text_preview = match.metadata.get("text", "")[:50].replace('\n', ' ')
                page_num = match.metadata.get("page_number", "N/A")
                logger.info(
                    f"Result {i+1}: ID={match.id}, score={match.score:.4f}, "
                    f"page={page_num}, text_preview='{text_preview}...'"
                )
            
            search_result = {
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results),
                "top_k": top_k,
                "filter": filter_dict
            }
            
            logger.info(f"=== VECTOR SEARCH END: {len(formatted_results)} results ===")
            return search_result
            
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            return {
                "query": query,
                "results": [],
                "total_results": 0,
                "error": str(e)
            }
    
    def create_metadata_filter(self, **kwargs) -> Dict[str, Any]:
        """
        Create metadata filter for Pinecone search
        
        Args:
            **kwargs: Filter parameters (author, chunk_type, page_number, etc.)
            
        Returns:
            Pinecone filter dictionary
        """
        filter_dict = {}
        
        # Author filter
        if "author" in kwargs and kwargs["author"]:
            filter_dict["authors.name"] = {"$eq": kwargs["author"]}
        
        # Page number filter
        if "page_number" in kwargs and kwargs["page_number"]:
            filter_dict["page_number"] = {"$eq": int(kwargs["page_number"])}
        
        # Document type filter
        if "document_type" in kwargs and kwargs["document_type"]:
            filter_dict["document_type"] = {"$eq": kwargs["document_type"]}
        
        # Text length filter
        if "min_text_length" in kwargs:
            filter_dict["text_length"] = {"$gte": int(kwargs["min_text_length"])}
        
        return filter_dict if filter_dict else None
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive vector store statistics
        
        Returns:
            Statistics dictionary
        """
        try:
            # Get Pinecone stats
            pinecone_stats = self.pinecone_manager.get_index_stats()
            
            stats = {
                "index_name": self.config.index_name,
                "dimension": self.config.dimension,
                "metric": self.config.metric,
                "embedding_model": self.config.embedding_model,
                "total_vectors": pinecone_stats.get("total_vector_count", 0),
                "index_fullness": pinecone_stats.get("index_fullness", 0),
                "namespaces": pinecone_stats.get("namespaces", {}),
                "configuration": {
                    "batch_size": self.config.batch_size,
                    "search_top_k": self.config.search_top_k,
                    "cloud": self.config.cloud,
                    "region": self.config.region
                }
            }
            
            logger.info(f"Vector store stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting vector store stats: {str(e)}")
            return {"error": str(e)}
    
    def delete_all_vectors(self) -> bool:
        """
        Delete all vectors from the index (use with caution)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.warning("Deleting ALL vectors from index")
            
            # Delete all vectors
            self.pinecone_manager.index.delete(delete_all=True)
            
            logger.info("All vectors deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting all vectors: {str(e)}")
            return False


class DocumentIndexer:
    """
    High-level document indexer that combines processing and vector indexing
    
    This class orchestrates the complete document indexing pipeline:
    - Document processing with LlamaParse
    - Metadata extraction with LlamaExtract
    - Chunking with hybrid strategy
    - Vector indexing with embeddings
    """
    
    def __init__(self, vector_store_config: Optional[VectorStoreConfig] = None):
        """
        Initialize document indexer
        
        Args:
            vector_store_config: Optional vector store configuration
        """
        self.vector_store = VectorStore(vector_store_config)
        logger.info("DocumentIndexer initialized")
    
    def index_processed_chunks(self, chunks: List[TextNode]) -> Dict[str, Any]:
        """
        Index pre-processed chunks
        
        Args:
            chunks: List of processed TextNode chunks
            
        Returns:
            Indexing results
        """
        try:
            logger.info(f"Indexing {len(chunks)} processed chunks")
            
            result = self.vector_store.index_chunks(chunks)
            
            logger.info(f"Chunk indexing completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error indexing processed chunks: {str(e)}")
            return {
                "indexed_count": 0,
                "total_chunks": len(chunks),
                "success": False,
                "error": str(e)
            }
    
    def search_documents(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Search indexed documents
        
        Args:
            query: Search query
            **kwargs: Additional search parameters
            
        Returns:
            Search results
        """
        # Create metadata filter if filter parameters provided
        filter_dict = self.vector_store.create_metadata_filter(**kwargs)
        
        return self.vector_store.search(
            query=query,
            top_k=kwargs.get("top_k", None),
            filter_dict=filter_dict
        )


# Convenience functions for easy usage
def create_vector_store(**kwargs) -> VectorStore:
    """
    Convenience function to create vector store with custom configuration
    
    Args:
        **kwargs: Configuration options
        
    Returns:
        VectorStore instance
    """
    config = VectorStoreConfig(**kwargs)
    return VectorStore(config)


def index_documents(documents: List[Union[Document, TextNode]], **kwargs) -> Dict[str, Any]:
    """
    Convenience function to index documents with default settings
    
    Args:
        documents: Documents to index
        **kwargs: Configuration options
        
    Returns:
        Indexing results
    """
    vector_store = create_vector_store(**kwargs)
    return vector_store.index_documents(documents)


def search_documents(query: str, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to search documents with default settings
    
    Args:
        query: Search query
        **kwargs: Configuration and filter options
        
    Returns:
        Search results
    """
    vector_store = create_vector_store(**kwargs)
    
    # Extract search parameters
    search_params = {
        "top_k": kwargs.pop("top_k", None),
        "filter_dict": vector_store.create_metadata_filter(**kwargs)
    }
    
    return vector_store.search(query, **search_params)