"""
Query engine module with Cohere reranking and GPT-4o-mini generation

This module provides complete query processing pipeline:
- Query enhancement (rewriting and expansion)
- Vector search with Pinecone
- Cohere reranking for top results
- Answer generation with GPT-4o-mini using specialized system prompt
Based on PRD specifications for Russian language output.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field
from llama_index.core import Document
from llama_index.core.schema import TextNode

from .config import settings, get_system_prompt
from .vector_store import VectorStore, VectorStoreConfig

# Set up logger
logger = logging.getLogger(__name__)


class QueryEngineConfig(BaseModel):
    """Configuration for QueryEngine"""
    
    # API configurations
    openai_api_key: str = Field(default_factory=lambda: settings.openai_api_key)
    cohere_api_key: str = Field(default_factory=lambda: settings.cohere_api_key)
    
    # Model configurations
    chat_model: str = Field(default=settings.openai_chat_model)  # gpt-4o-mini
    cohere_model: str = Field(default=settings.cohere_rerank_model)  # rerank-english-v2.0
    
    # Query processing settings
    search_top_k: int = Field(default=settings.search_top_k)  # 50 candidates from Pinecone
    rerank_top_n: int = Field(default=settings.rerank_top_n)  # 10 final documents
    answer_temperature: float = Field(default=settings.answer_temperature)  # 0.1
    
    # Query enhancement settings
    enable_query_rewriting: bool = Field(default=True)
    enable_query_expansion: bool = Field(default=True)
    max_query_length: int = Field(default=500)
    
    # Answer generation settings
    max_context_length: int = Field(default=8000)  # Characters in context
    include_sources: bool = Field(default=True)
    response_language: str = Field(default="russian")


class QueryEnhancer:
    """
    Query enhancement with rewriting and expansion
    """
    
    def __init__(self, config: QueryEngineConfig):
        """
        Initialize query enhancer
        
        Args:
            config: Query engine configuration
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
        
        logger.info("QueryEnhancer initialized")
    
    def rewrite_query(self, query: str) -> str:
        """
        Rewrite query to improve search results
        
        Args:
            query: Original query
            
        Returns:
            Rewritten query
        """
        try:
            if not self.config.enable_query_rewriting:
                return query
            
            logger.info("Rewriting query for better search")
            
            system_prompt = """You are a query rewriting system for scientific papers in economics, quantum finance, and machine learning. 

Your task is to rewrite the user's query to improve search results in a scientific document database.

Guidelines:
- Keep the core meaning of the original query
- Use more specific scientific terminology
- Add relevant domain-specific keywords
- Make the query more precise and searchable
- Keep it concise and focused
- Return only the rewritten query without explanations"""
            
            response = self.client.chat.completions.create(
                model=self.config.chat_model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Original query: {query}"}
                ]
            )
            
            rewritten_query = response.choices[0].message.content.strip()
            
            # Validate length
            if len(rewritten_query) > self.config.max_query_length:
                rewritten_query = rewritten_query[:self.config.max_query_length].rsplit(' ', 1)[0]
            
            logger.info(f"Query rewritten: '{query}' -> '{rewritten_query}'")
            return rewritten_query
            
        except Exception as e:
            logger.error(f"Error rewriting query: {str(e)}")
            return query  # Return original query on error
    
    def expand_query(self, query: str) -> str:
        """
        Expand query with synonyms and related terms
        
        Args:
            query: Query to expand
            
        Returns:
            Expanded query
        """
        try:
            if not self.config.enable_query_expansion:
                return query
            
            logger.info("Expanding query with related terms")
            
            system_prompt = """You are a query expansion system for scientific papers in economics, quantum finance, and machine learning.

Your task is to add relevant synonyms and related terms to the user's query to improve search recall.

Guidelines:
- Add 3-5 relevant synonyms or related terms
- Focus on scientific terminology
- Include alternative spellings or formulations
- Use terms commonly found in academic papers
- Separate terms with commas
- Return the expanded terms without the original query"""
            
            response = self.client.chat.completions.create(
                model=self.config.chat_model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query to expand: {query}"}
                ]
            )
            
            expansion_terms = response.choices[0].message.content.strip()
            
            # Combine original query with expansion terms
            expanded_query = f"{query}, {expansion_terms}"
            
            # Validate length
            if len(expanded_query) > self.config.max_query_length:
                expanded_query = expanded_query[:self.config.max_query_length].rsplit(',', 1)[0]
            
            logger.info(f"Query expanded with terms: {expansion_terms}")
            return expanded_query
            
        except Exception as e:
            logger.error(f"Error expanding query: {str(e)}")
            return query  # Return original query on error
    
    def enhance_query(self, query: str) -> Dict[str, str]:
        """
        Enhance query with both rewriting and expansion
        
        Args:
            query: Original query
            
        Returns:
            Dictionary with original, rewritten, and enhanced queries
        """
        try:
            logger.info(f"Enhancing query: {query}")
            
            # Step 1: Rewrite query
            rewritten_query = self.rewrite_query(query)
            
            # Step 2: Expand the rewritten query
            enhanced_query = self.expand_query(rewritten_query)
            
            result = {
                "original": query,
                "rewritten": rewritten_query,
                "enhanced": enhanced_query
            }
            
            logger.info("Query enhancement completed")
            return result
            
        except Exception as e:
            logger.error(f"Error enhancing query: {str(e)}")
            return {
                "original": query,
                "rewritten": query,
                "enhanced": query
            }


class CohereReranker:
    """
    Cohere reranking for improved result relevance
    """
    
    def __init__(self, config: QueryEngineConfig):
        """
        Initialize Cohere reranker
        
        Args:
            config: Query engine configuration
        """
        self.config = config
        
        # Validate API key
        if not self.config.cohere_api_key or self.config.cohere_api_key.startswith("your_"):
            raise ValueError(
                "COHERE_API_KEY is required. "
                "Set it in environment variables or .env file"
            )
        
        # Initialize Cohere client
        try:
            import cohere
            self.client = cohere.Client(self.config.cohere_api_key)
        except ImportError:
            raise ImportError("Cohere library not found. Install with: pip install cohere")
        
        logger.info(f"CohereReranker initialized with model: {self.config.cohere_model}")
    
    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank documents using Cohere Rerank
        
        Args:
            query: Search query
            documents: List of document dictionaries from vector search
            
        Returns:
            Reranked documents with relevance scores
        """
        try:
            if not documents:
                return []
            
            logger.info(f"=== COHERE RERANK START ===")
            logger.info(f"Query: {query[:100]}...")
            logger.info(f"Input documents: {len(documents)}")
            
            # Check for duplicates in input documents
            input_ids = [doc.get("id", "NO_ID") for doc in documents]
            unique_input_ids = set(input_ids)
            if len(unique_input_ids) != len(input_ids):
                duplicates = [id for id in input_ids if input_ids.count(id) > 1]
                logger.warning(f"DUPLICATES FOUND in rerank INPUT: {set(duplicates)}")
                logger.warning(f"Total input docs: {len(input_ids)}, Unique IDs: {len(unique_input_ids)}")
            else:
                logger.info(f"No duplicates in rerank input - all {len(input_ids)} docs have unique IDs")
            
            # Prepare documents for Cohere
            doc_texts = []
            doc_metadata = []
            
            for i, doc in enumerate(documents):
                # Extract text for reranking
                text = doc.get("text", "")
                if not text and "metadata" in doc:
                    text = doc["metadata"].get("text", "")
                
                doc_texts.append(text[:1000])  # Limit text length for Cohere
                doc_metadata.append(doc)
                
                # Log each input document
                doc_id = doc.get("id", "NO_ID")
                original_score = doc.get("score", 0)
                text_preview = text[:50].replace('\n', ' ')
                logger.info(
                    f"Input doc {i+1}: ID={doc_id}, original_score={original_score:.4f}, "
                    f"text_preview='{text_preview}...'"
                )
            
            if not doc_texts:
                logger.warning("No valid texts found for reranking")
                return documents[:self.config.rerank_top_n]
            
            # Perform reranking
            logger.info(f"Sending to Cohere: {len(doc_texts)} docs, model={self.config.cohere_model}")
            rerank_response = self.client.rerank(
                model=self.config.cohere_model,
                query=query,
                documents=doc_texts,
                top_n=self.config.rerank_top_n
            )
            
            logger.info(f"Cohere returned {len(rerank_response.results)} reranked results")
            
            # Process reranked results
            reranked_documents = []
            output_ids = []
            
            for i, result in enumerate(rerank_response.results):
                original_doc = doc_metadata[result.index]
                
                # Add Cohere relevance score
                reranked_doc = {
                    **original_doc,
                    "cohere_score": result.relevance_score,
                    "original_rank": result.index,
                    "reranked_position": len(reranked_documents) + 1
                }
                
                reranked_documents.append(reranked_doc)
                output_ids.append(original_doc.get("id", "NO_ID"))
                
                # Log each reranked result
                doc_id = original_doc.get("id", "NO_ID")
                original_score = original_doc.get("score", 0)
                logger.info(
                    f"Reranked {i+1}: ID={doc_id}, cohere_score={result.relevance_score:.4f}, "
                    f"original_score={original_score:.4f}, original_rank={result.index}"
                )
            
            # Check for duplicates in output
            unique_output_ids = set(output_ids)
            if len(unique_output_ids) != len(output_ids):
                duplicates = [id for id in output_ids if output_ids.count(id) > 1]
                logger.warning(f"DUPLICATES FOUND in rerank OUTPUT: {set(duplicates)}")
                logger.warning(f"Total output docs: {len(output_ids)}, Unique IDs: {len(unique_output_ids)}")
            else:
                logger.info(f"No duplicates in rerank output - all {len(output_ids)} docs have unique IDs")
            
            logger.info(f"=== COHERE RERANK END: {len(reranked_documents)} documents ===")
            return reranked_documents
            
        except Exception as e:
            logger.error(f"Error reranking documents: {str(e)}")
            # Fallback: return top N documents from original search
            return documents[:self.config.rerank_top_n]


class AnswerGenerator:
    """
    Answer generation using GPT-4o-mini with specialized system prompt
    """
    
    def __init__(self, config: QueryEngineConfig):
        """
        Initialize answer generator
        
        Args:
            config: Query engine configuration
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
        
        # Load specialized system prompt from config
        self.system_prompt = get_system_prompt()
        
        logger.info("AnswerGenerator initialized with specialized system prompt")
    
    def generate_answer(self, query: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate answer using GPT-4o-mini with specialized prompt
        
        Args:
            query: User query
            documents: Reranked documents with context
            
        Returns:
            Generated answer with metadata
        """
        try:
            if not documents:
                return {
                    "answer": "Извините, не найдено релевантных документов для ответа на ваш вопрос.",
                    "has_sources": False,
                    "source_count": 0
                }
            
            logger.info(f"=== ANSWER GENERATION START ===")
            logger.info(f"Query: {query[:100]}...")
            logger.info(f"Input documents: {len(documents)}")
            
            # Check for duplicates in input documents
            input_ids = [doc.get("id", "NO_ID") for doc in documents]
            unique_input_ids = set(input_ids)
            if len(unique_input_ids) != len(input_ids):
                duplicates = [id for id in input_ids if input_ids.count(id) > 1]
                logger.warning(f"DUPLICATES FOUND in answer generation INPUT: {set(duplicates)}")
                logger.warning(f"Total input docs: {len(input_ids)}, Unique IDs: {len(unique_input_ids)}")
                
                # Show which documents are duplicates
                for duplicate_id in set(duplicates):
                    duplicate_positions = [i for i, id in enumerate(input_ids) if id == duplicate_id]
                    logger.warning(f"Document ID '{duplicate_id}' appears at positions: {duplicate_positions}")
            else:
                logger.info(f"No duplicates in answer generation input - all {len(input_ids)} docs have unique IDs")
            
            # Prepare context from documents
            context_parts = []
            sources = []
            total_text_length = 0
            
            for i, doc in enumerate(documents):
                # Extract document information
                text = doc.get("text", "")
                metadata = doc.get("metadata", {})
                cohere_score = doc.get("cohere_score", 0)
                original_score = doc.get("score", 0)
                
                # Log each document used for answer generation
                doc_id = doc.get("id", "NO_ID")
                text_preview = text[:50].replace('\n', ' ')
                page_num = metadata.get("page_number", "N/A")
                logger.info(
                    f"Answer doc {i+1}: ID={doc_id}, cohere_score={cohere_score:.4f}, "
                    f"original_score={original_score:.4f}, page={page_num}, "
                    f"text_len={len(text)}, preview='{text_preview}...'"
                )
                
                # Create context entry
                context_entry = f"Документ {i+1}:\n"
                context_entry += f"Текст: {text}\n"
                
                # Add metadata if available
                if metadata:
                    if "title" in metadata:
                        context_entry += f"Название: {metadata['title']}\n"
                    if "authors" in metadata:
                        authors = metadata['authors']
                        if isinstance(authors, list):
                            author_names = [author.get('name', '') if isinstance(author, dict) else str(author) for author in authors]
                            context_entry += f"Авторы: {', '.join(author_names)}\n"
                    if "page_number" in metadata:
                        context_entry += f"Страница: {metadata['page_number']}\n"
                    if "section_title" in metadata:
                        context_entry += f"Раздел: {metadata['section_title']}\n"
                
                context_entry += f"Релевантность: {cohere_score:.3f}\n"
                context_parts.append(context_entry)
                total_text_length += len(text)
                
                # Prepare source info
                source_info = {
                    "document_id": doc.get("id", f"doc_{i+1}"),
                    "text": text[:200] + "..." if len(text) > 200 else text,
                    "metadata": metadata,
                    "relevance_score": cohere_score,
                    "rank": i + 1
                }
                sources.append(source_info)
            
            # Combine context
            full_context = "\n\n".join(context_parts)
            original_context_length = len(full_context)
            
            # Limit context length
            context_truncated = False
            if len(full_context) > self.config.max_context_length:
                full_context = full_context[:self.config.max_context_length] + "\n...[контекст обрезан]"
                context_truncated = True
            
            logger.info(f"Context stats: {len(documents)} docs, {total_text_length} chars total text, "
                       f"{original_context_length} chars context, truncated={context_truncated}")
            
            # Generate answer
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Вопрос: {query}\n\nКонтекст:\n{full_context}"}
            ]
            
            logger.info(f"Sending to GPT-4o-mini: {len(messages)} messages, "
                       f"context_length={len(full_context)}, model={self.config.chat_model}")
            
            response = self.client.chat.completions.create(
                model=self.config.chat_model,
                temperature=self.config.answer_temperature,
                messages=messages
            )
            
            answer = response.choices[0].message.content
            
            result = {
                "answer": answer,
                "has_sources": len(sources) > 0,
                "source_count": len(sources),
                "sources": sources if self.config.include_sources else [],
                "model_used": self.config.chat_model,
                "temperature": self.config.answer_temperature
            }
            
            logger.info(f"Answer generated: {len(answer)} chars, {len(sources)} sources")
            logger.info(f"=== ANSWER GENERATION END ===")
            return result
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return {
                "answer": f"Произошла ошибка при генерации ответа: {str(e)}",
                "has_sources": False,
                "source_count": 0,
                "error": str(e)
            }


class QueryEngine:
    """
    Complete query processing engine
    
    This class orchestrates the entire query processing pipeline:
    1. Query enhancement (rewriting and expansion)  
    2. Vector search with Pinecone (50 candidates)
    3. Cohere reranking (top 10 documents)
    4. Answer generation with GPT-4o-mini (Russian output)
    
    Based on PRD specifications for ArXiv scientific document analysis.
    """
    
    def __init__(self, config: Optional[QueryEngineConfig] = None,
                 vector_store_config: Optional[VectorStoreConfig] = None):
        """
        Initialize query engine
        
        Args:
            config: Optional query engine configuration
            vector_store_config: Optional vector store configuration
        """
        self.config = config or QueryEngineConfig()
        
        # Initialize components
        self.query_enhancer = QueryEnhancer(self.config)
        self.vector_store = VectorStore(vector_store_config)
        self.reranker = CohereReranker(self.config)
        self.answer_generator = AnswerGenerator(self.config)
        
        logger.info("QueryEngine initialized with all components")
    
    def process_query(self, query: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process query end-to-end
        
        Args:
            query: User query
            filters: Optional metadata filters
            
        Returns:
            Complete query result with answer and sources
        """
        try:
            start_time = time.time()
            logger.info(f"======== QUERY PROCESSING START ========")
            logger.info(f"Query: {query}")
            logger.info(f"Filters: {filters}")
            
            # Step 1: Query enhancement
            logger.info(f"STEP 1: Query enhancement")
            enhanced_queries = self.query_enhancer.enhance_query(query)
            search_query = enhanced_queries["enhanced"]
            logger.info(f"Enhanced query: {search_query}")
            
            # Step 2: Vector search
            logger.info(f"STEP 2: Vector search (target: {self.config.search_top_k} docs)")
            search_filter = None
            if filters:
                search_filter = self.vector_store.create_metadata_filter(**filters)
            
            search_results = self.vector_store.search(
                query=search_query,
                top_k=self.config.search_top_k,
                filter_dict=search_filter
            )
            
            if not search_results.get("results"):
                logger.warning("No search results found")
                logger.info(f"======== QUERY PROCESSING END (No results) ========")
                return {
                    "query": query,
                    "enhanced_queries": enhanced_queries,
                    "answer": "Не найдено релевантных документов для ответа на ваш вопрос.",
                    "sources": [],
                    "search_results_count": 0,
                    "reranked_results_count": 0,
                    "processing_time": time.time() - start_time,
                    "success": True
                }
            
            logger.info(f"Vector search completed: {len(search_results['results'])} documents")
            
            # Step 3: Reranking with Cohere
            logger.info(f"STEP 3: Cohere reranking (target: {self.config.rerank_top_n} docs)")
            reranked_documents = self.reranker.rerank(
                query=query,  # Use original query for reranking
                documents=search_results["results"]
            )
            
            logger.info(f"Reranking completed: {len(reranked_documents)} documents")
            
            # Step 4: Answer generation
            logger.info(f"STEP 4: Answer generation")
            answer_result = self.answer_generator.generate_answer(
                query=query,
                documents=reranked_documents
            )
            
            # Compile final result
            result = {
                "query": query,
                "enhanced_queries": enhanced_queries,
                "answer": answer_result.get("answer", ""),
                "sources": answer_result.get("sources", []),
                "search_results_count": len(search_results.get("results", [])),
                "reranked_results_count": len(reranked_documents),
                "processing_time": time.time() - start_time,
                "success": True,
                "filters_applied": filters,
                "model_info": {
                    "chat_model": self.config.chat_model,
                    "rerank_model": self.config.cohere_model,
                    "embedding_model": self.vector_store.config.embedding_model
                }
            }
            
            logger.info(f"FINAL RESULT: search={result['search_results_count']}, "
                       f"reranked={result['reranked_results_count']}, "
                       f"sources={len(result.get('sources', []))}, "
                       f"time={result['processing_time']:.2f}s")
            logger.info(f"======== QUERY PROCESSING END (Success) ========")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "query": query,
                "enhanced_queries": {"original": query, "rewritten": query, "enhanced": query},
                "answer": f"Произошла ошибка при обработке запроса: {str(e)}",
                "sources": [],
                "search_results_count": 0,
                "reranked_results_count": 0,
                "processing_time": time.time() - start_time if 'start_time' in locals() else 0,
                "success": False,
                "error": str(e)
            }
    
    def batch_process_queries(self, queries: List[str], 
                             filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch
        
        Args:
            queries: List of queries to process
            filters: Optional metadata filters
            
        Returns:
            List of query results
        """
        results = []
        
        for i, query in enumerate(queries):
            logger.info(f"Processing query {i+1}/{len(queries)}")
            
            result = self.process_query(query, filters)
            result["batch_index"] = i
            results.append(result)
            
            # Add small delay between queries to avoid rate limiting
            if i < len(queries) - 1:
                time.sleep(0.5)
        
        logger.info(f"Batch processing completed: {len(results)} queries")
        return results
    
    def get_query_stats(self) -> Dict[str, Any]:
        """
        Get query engine statistics
        
        Returns:
            Statistics dictionary
        """
        vector_stats = self.vector_store.get_vector_store_stats()
        
        stats = {
            "query_engine_config": {
                "chat_model": self.config.chat_model,
                "rerank_model": self.config.cohere_model,
                "search_top_k": self.config.search_top_k,
                "rerank_top_n": self.config.rerank_top_n,
                "answer_temperature": self.config.answer_temperature
            },
            "vector_store_stats": vector_stats,
            "features": {
                "query_rewriting": self.config.enable_query_rewriting,
                "query_expansion": self.config.enable_query_expansion,
                "cohere_reranking": True,
                "russian_output": True
            }
        }
        
        return stats


# Convenience functions for easy usage
def create_query_engine(**kwargs) -> QueryEngine:
    """
    Convenience function to create query engine with custom configuration
    
    Args:
        **kwargs: Configuration options
        
    Returns:
        QueryEngine instance
    """
    # Separate query engine and vector store configs
    query_config_keys = {
        'openai_api_key', 'cohere_api_key', 'chat_model', 'cohere_model',
        'search_top_k', 'rerank_top_n', 'answer_temperature',
        'enable_query_rewriting', 'enable_query_expansion', 'max_query_length',
        'max_context_length', 'include_sources', 'response_language'
    }
    
    query_config = {k: v for k, v in kwargs.items() if k in query_config_keys}
    vector_config = {k: v for k, v in kwargs.items() if k not in query_config_keys}
    
    query_engine_config = QueryEngineConfig(**query_config)
    vector_store_config = VectorStoreConfig(**vector_config) if vector_config else None
    
    return QueryEngine(query_engine_config, vector_store_config)


def process_query(query: str, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to process single query with default settings
    
    Args:
        query: Query to process
        **kwargs: Configuration and filter options
        
    Returns:
        Query result
    """
    # Extract filters
    filters = {k: v for k, v in kwargs.items() 
              if k in ['author', 'page_number', 'document_type']}
    
    # Remove filters from kwargs for engine config
    engine_kwargs = {k: v for k, v in kwargs.items() if k not in filters}
    
    query_engine = create_query_engine(**engine_kwargs)
    return query_engine.process_query(query, filters if filters else None)