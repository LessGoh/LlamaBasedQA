"""
Streamlit application for ArXiv RAG System

This is the main Streamlit application that provides a user interface for:
- PDF document upload and processing
- Search interface with filters
- Query history and caching
- Results display with sources and citations

Based on PRD specifications for ArXiv scientific document analysis.
"""

import streamlit as st
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
try:
    from src.document_processor import ArxivDocumentProcessor
    from src.metadata_extractor import MetadataExtractor  
    from src.chunking_simplified import HybridChunker
    from src.vector_store import DocumentIndexer
    from src.query_engine import QueryEngine
    from src.utils import QueryCache, ResultFormatter, FileValidator, ConfigValidator, metrics, format_processing_time
    from src.config import validate_api_keys
    from llama_index.core import Document
    
    # Import new parallel processing components
    from src.parallel_workflow_manager import ParallelWorkflowManager, BatchProcessingConfig, process_documents_batch
    from src.workflow_events import ProcessingStatus
except ImportError as e:
    st.error(f"Error importing modules: {str(e)}")
    st.stop()

# Set page config
st.set_page_config(
    page_title="ArXiv RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .stAlert {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    .source-card {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    
    .metric-card {
        background-color: #f1f3f4;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    
    .query-stats {
        font-size: 0.8rem;
        color: #6c757d;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_components():
    """Initialize and cache system components"""
    try:
        logger.info("Initializing RAG system components...")
        
        # Check API keys first
        missing_keys = ConfigValidator.get_missing_api_keys()
        if missing_keys:
            st.error(f"Missing API keys: {', '.join(missing_keys)}")
            st.info("Please set the required API keys in your .env file")
            st.stop()
        
        # Initialize components
        doc_processor = ArxivDocumentProcessor()
        metadata_processor = MetadataExtractor()
        chunker = HybridChunker()
        indexer = DocumentIndexer()
        query_engine = QueryEngine()
        query_cache = QueryCache(ttl_seconds=3600, max_size=50)
        
        logger.info("All components initialized successfully")
        
        return {
            "doc_processor": doc_processor,
            "metadata_processor": metadata_processor,
            "chunker": chunker,
            "indexer": indexer,
            "query_engine": query_engine,
            "query_cache": query_cache
        }
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        st.error(f"Failed to initialize system: {str(e)}")
        st.stop()


def process_uploaded_pdf(uploaded_file, components: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process uploaded PDF file"""
    try:
        # Validate file
        file_content = uploaded_file.getvalue()
        
        if not FileValidator.validate_pdf(file_content):
            st.error("Invalid PDF file. Please upload a valid PDF document.")
            return None
        
        file_size_mb = FileValidator.get_file_size_mb(file_content)
        if not FileValidator.validate_file_size(file_content, max_size_mb=50):
            st.error(f"File too large ({file_size_mb:.1f}MB). Maximum size is 50MB.")
            return None
        
        st.info(f"Processing PDF file ({file_size_mb:.1f}MB)...")
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        start_time = time.time()
        
        # Step 1: Process PDF with LlamaParse
        st.info("🔄 Extracting text from PDF...")
        doc_result = components["doc_processor"].process(tmp_file_path)
        
        if not doc_result["success"]:
            st.error(f"Failed to process PDF: {doc_result.get('error', 'Unknown error')}")
            return None
        
        documents = doc_result["documents"]
        
        # Step 2: Extract metadata
        st.info("🔄 Extracting metadata...")
        # Объединяем все страницы в один полный документ для корректной обработки
        full_text = "\n\n".join([doc.text for doc in documents])
        full_document = Document(text=full_text, metadata=documents[0].metadata)
        metadata_result = components["metadata_processor"].extract_metadata(full_document)
        
        # Convert ExtractedMetadata object to dictionary for compatibility
        metadata_dict = metadata_result.model_dump() if hasattr(metadata_result, 'model_dump') else metadata_result
        
        # Step 3: Chunk documents
        st.info("🔄 Creating document chunks...")
        chunks = components["chunker"].chunk_document(full_document, metadata_dict)
        
        # Step 4: Index chunks
        st.info("🔄 Indexing in vector database...")
        index_result = components["indexer"].index_processed_chunks(chunks)
        
        if not index_result["success"]:
            st.error(f"Failed to index document: {index_result.get('error', 'Unknown error')}")
            return None
        
        processing_time = time.time() - start_time
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        # Record metrics
        metrics.record_timing("document_processing", processing_time)
        metrics.increment_counter("documents_processed")
        
        result = {
            "filename": uploaded_file.name,
            "processing_time": processing_time,
            "metadata": metadata_dict,
            "chunks_created": len(chunks),
            "chunks_indexed": index_result["indexed_count"],
            "success": True
        }
        
        logger.info(f"PDF processing completed in {processing_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        st.error(f"Error processing PDF: {str(e)}")
        return None


def display_document_metadata(metadata: Dict[str, Any]):
    """Display extracted document metadata"""
    st.subheader("📄 Document Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Title:**")
        st.write(metadata.get("title", "Unknown"))
        
        st.markdown("**Authors:**")
        authors = metadata.get("authors", [])
        if authors:
            for author in authors:
                if isinstance(author, dict):
                    name = author.get("name", "Unknown")
                    affiliation = author.get("affiliation", "")
                    if affiliation:
                        st.write(f"• {name} ({affiliation})")
                    else:
                        st.write(f"• {name}")
                else:
                    st.write(f"• {author}")
        else:
            st.write("No authors found")
        
        st.markdown("**Domain:**")
        st.write(metadata.get("domain", "Unknown"))
    
    with col2:
        st.markdown("**Abstract:**")
        abstract = metadata.get("abstract", "")
        if abstract:
            st.write(abstract[:500] + ("..." if len(abstract) > 500 else ""))
        else:
            st.write("No abstract found")
        
        st.markdown("**Main Findings:**")
        findings = metadata.get("mainFindings", [])
        if findings:
            for finding in findings[:3]:  # Show first 3 findings
                st.write(f"• {finding}")
        else:
            st.write("No main findings extracted")
        
        st.markdown("**Complexity Score:**")
        complexity = metadata.get("complexity_score", 0)
        st.progress(complexity)
        st.write(f"{complexity:.2f}/1.0")


def create_search_interface(components: Dict[str, Any]):
    """Create the search interface"""
    st.header("🔍 Search Documents")
    
    # Search input
    query = st.text_area(
        "Enter your question about economics, quantum finance, or machine learning:",
        height=100,
        placeholder="Example: Какие методы используются для оценки волатильности в квантовых финансах?"
    )
    
    # Advanced search options
    with st.expander("⚙️ Advanced Search Options"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            author_filter = st.text_input("Author name")
        
        with col2:
            page_number = st.number_input("Page number", min_value=0, value=0)
        
        with col3:
            top_k = st.slider("Number of results", min_value=5, max_value=20, value=10)
    
    # Search button
    search_button = st.button("🔍 Search", type="primary")
    
    if search_button and query.strip():
        # Prepare filters
        filters = {}
        if author_filter:
            filters["author"] = author_filter
        if page_number > 0:
            filters["page_number"] = page_number
        
        # Check cache first
        cache_key = f"{query}:{str(filters) if filters else None}"
        cached_result = components["query_cache"].get(query, str(filters) if filters else None)
        
        if cached_result:
            st.info("📋 Result retrieved from cache")
            display_search_results(cached_result)
        else:
            # Perform search
            with st.spinner("🔄 Processing query..."):
                start_time = time.time()
                
                try:
                    # Add top_k to filters for query engine
                    search_filters = {**filters, "top_k": top_k}
                    
                    result = components["query_engine"].process_query(query, search_filters)
                    
                    # Format result for display
                    formatted_result = ResultFormatter.format_answer_for_display(result)
                    
                    # Cache result
                    components["query_cache"].set(query, str(filters) if filters else None, formatted_result)
                    
                    # Record metrics
                    processing_time = time.time() - start_time
                    metrics.record_timing("query_processing", processing_time)
                    metrics.increment_counter("queries_processed")
                    
                    # Display results
                    display_search_results(formatted_result)
                    
                except Exception as e:
                    logger.error(f"Error processing query: {str(e)}")
                    st.error(f"Error processing query: {str(e)}")


def display_search_results(result: Dict[str, Any]):
    """Display search results"""
    if not result.get("success", True):
        st.error(f"Query failed: {result.get('error', 'Unknown error')}")
        return
    
    # Display answer
    st.subheader("💡 Answer")
    answer = result.get("answer", "")
    if answer:
        st.markdown(answer)
    else:
        st.warning("No answer generated")
    
    # Display query statistics
    stats = result.get("stats", {})
    processing_time = result.get("processing_time", 0)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Search Results", stats.get("search_results", 0))
    with col2:
        st.metric("Reranked Results", stats.get("reranked_results", 0))
    with col3:
        st.metric("Sources Used", stats.get("source_count", 0))
    with col4:
        st.metric("Processing Time", format_processing_time(processing_time))
    
    # Display enhanced queries if available
    if "enhanced_queries" in result:
        with st.expander("🔄 Query Enhancement"):
            enhanced = result["enhanced_queries"]
            st.write("**Original:**", enhanced.get("original", ""))
            st.write("**Rewritten:**", enhanced.get("rewritten", ""))
            st.write("**Enhanced:**", enhanced.get("enhanced", ""))
    
    # Display sources
    sources = result.get("sources", [])
    if sources:
        st.subheader("📚 Sources")
        
        for source in sources:
            with st.expander(f"Source {source['rank']} (Relevance: {source['relevance_score']:.3f})"):
                metadata = source["metadata"]
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**Text:**")
                    st.write(source["text"])
                    
                with col2:
                    st.markdown("**Metadata:**")
                    st.write(f"**Title:** {metadata.get('title', 'Unknown')}")
                    
                    authors = metadata.get('authors', [])
                    if authors:
                        if isinstance(authors, list) and len(authors) > 0:
                            if isinstance(authors[0], dict):
                                author_names = [author.get('name', 'Unknown') for author in authors]
                            else:
                                author_names = [str(author) for author in authors]
                            st.write(f"**Authors:** {', '.join(author_names[:3])}")
                    
                    if metadata.get('page', 0) > 0:
                        st.write(f"**Page:** {metadata['page']}")
                    
                    if metadata.get('section'):
                        st.write(f"**Section:** {metadata['section']}")
        
        # Display citation format
        with st.expander("📖 Citation Format"):
            citation_text = ResultFormatter.format_sources_for_citation(sources)
            st.code(citation_text, language="text")
    else:
        st.warning("No sources found for this query")


def create_history_interface(components: Dict[str, Any]):
    """Create query history interface"""
    st.header("📝 Query History")
    
    # Get query history
    history = components["query_cache"].get_history()
    
    if not history:
        st.info("No previous queries found")
        return
    
    st.write(f"Found {len(history)} previous queries:")
    
    # Display queries with option to rerun
    for i, past_query in enumerate(reversed(history[-10:])):  # Show last 10 queries
        with st.container():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.write(f"**{i+1}.** {past_query}")
            
            with col2:
                if st.button("🔄 Rerun", key=f"rerun_{i}"):
                    # Set query in session state and switch to search tab
                    st.session_state.selected_query = past_query
                    st.rerun()
    
    # Clear history option
    if st.button("🗑️ Clear History", type="secondary"):
        components["query_cache"].clear()
        st.success("Query history cleared")
        st.rerun()


def create_statistics_interface(components: Dict[str, Any]):
    """Create system statistics interface"""
    st.header("📊 System Statistics")
    
    try:
        # Query engine stats
        query_stats = components["query_engine"].get_query_stats()
        
        # Cache stats  
        cache_stats = components["query_cache"].get_stats()
        
        # System metrics
        system_metrics = metrics.get_stats()
        
        # Display in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 Query Engine")
            st.json(query_stats.get("query_engine_config", {}))
            
            st.subheader("📋 Cache Statistics")
            st.metric("Active Items", cache_stats.get("active_items", 0))
            st.metric("Total Items", cache_stats.get("total_items", 0))
            st.metric("Cache Hit Rate", f"{cache_stats.get('active_items', 0) / max(cache_stats.get('total_items', 1), 1) * 100:.1f}%")
        
        with col2:
            st.subheader("🏪 Vector Store")
            vector_stats = query_stats.get("vector_store_stats", {})
            if vector_stats:
                st.metric("Total Vectors", vector_stats.get("total_vectors", 0))
                st.metric("Index Name", vector_stats.get("index_name", "Unknown"))
                st.metric("Dimension", vector_stats.get("dimension", 0))
                st.metric("Embedding Model", vector_stats.get("embedding_model", "Unknown"))
            
            st.subheader("⏱️ Performance Metrics")
            if system_metrics.get("counters"):
                for counter, value in system_metrics["counters"].items():
                    st.metric(counter.replace("_", " ").title(), value)
        
        # Detailed metrics
        if system_metrics.get("timings"):
            st.subheader("⏱️ Timing Statistics")
            timing_data = []
            for operation, stats in system_metrics["timings"].items():
                timing_data.append({
                    "Operation": operation.replace("_", " ").title(),
                    "Count": stats["count"],
                    "Avg (s)": f"{stats['avg']:.2f}",
                    "Min (s)": f"{stats['min']:.2f}",
                    "Max (s)": f"{stats['max']:.2f}"
                })
            
            if timing_data:
                st.table(timing_data)
        
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        st.error(f"Error retrieving statistics: {str(e)}")


def create_batch_processing_interface(components: Dict[str, Any]):
    """Create the batch processing interface"""
    st.header("🔄 Batch Document Processing")
    st.markdown("""
    **Параллельная обработка множественных документов** - загрузите до 20 PDF файлов 
    для одновременной обработки. Система будет обрабатывать документы параллельно, 
    обеспечивая оптимальное использование ресурсов.
    """)
    
    # File upload section
    st.subheader("📂 Upload Files")
    uploaded_files = st.file_uploader(
        "Choose PDF files for batch processing",
        type="pdf",
        accept_multiple_files=True,
        help="Select multiple PDF files (up to 20) for parallel processing. Maximum size per file: 50MB"
    )
    
    if uploaded_files:
        st.info(f"✅ Selected {len(uploaded_files)} files for processing")
        
        # Validate files
        valid_files = []
        total_size = 0
        
        for uploaded_file in uploaded_files:
            file_size_mb = FileValidator.get_file_size_mb(uploaded_file.getvalue())
            total_size += file_size_mb
            
            if not FileValidator.validate_pdf(uploaded_file.getvalue()):
                st.error(f"❌ Invalid PDF file: {uploaded_file.name}")
                continue
                
            if not FileValidator.validate_file_size(uploaded_file.getvalue(), max_size_mb=50):
                st.error(f"❌ File too large: {uploaded_file.name} ({file_size_mb:.1f}MB > 50MB)")
                continue
                
            valid_files.append({
                "file": uploaded_file,
                "size_mb": file_size_mb,
                "name": uploaded_file.name
            })
        
        if len(uploaded_files) > 20:
            st.error(f"❌ Too many files: {len(uploaded_files)}. Maximum is 20 files per batch.")
            return
        
        if valid_files:
            st.success(f"✅ {len(valid_files)} valid files ready for processing (Total size: {total_size:.1f}MB)")
            
            # Display file list
            with st.expander("📄 File Details"):
                for i, file_info in enumerate(valid_files):
                    st.write(f"{i+1}. **{file_info['name']}** - {file_info['size_mb']:.1f}MB")
            
            # Configuration section
            with st.expander("⚙️ Processing Configuration"):
                col1, col2 = st.columns(2)
                
                with col1:
                    max_parallel = st.slider(
                        "Maximum Parallel Workflows",
                        min_value=1,
                        max_value=4,
                        value=2,
                        help="Number of documents to process simultaneously"
                    )
                    
                with col2:
                    workflow_timeout = st.slider(
                        "Workflow Timeout (minutes)",
                        min_value=10,
                        max_value=60,
                        value=30,
                        help="Maximum time to process each document"
                    )
            
            # Start processing button
            if not st.session_state.get('batch_processing_active', False):
                if st.button("🚀 Start Batch Processing", type="primary", key="start_batch"):
                    try:
                        # Save files to temporary directory
                        import tempfile
                        temp_dir = tempfile.mkdtemp()
                        file_paths = []
                        
                        for file_info in valid_files:
                            temp_path = os.path.join(temp_dir, file_info['name'])
                            with open(temp_path, 'wb') as f:
                                f.write(file_info['file'].getvalue())
                            file_paths.append(temp_path)
                        
                        # Initialize batch processing
                        config = BatchProcessingConfig(
                            max_parallel_workflows=max_parallel,
                            workflow_timeout=workflow_timeout * 60,  # Convert to seconds
                            queue_max_size=20
                        )
                        
                        # Store configuration in session state
                        st.session_state.batch_processing_active = True
                        st.session_state.batch_config = config
                        st.session_state.batch_file_paths = file_paths
                        st.session_state.batch_temp_dir = temp_dir
                        st.session_state.batch_indexer = components["indexer"]
                        
                        # Initialize processing status
                        st.session_state.batch_status = {path: "PENDING" for path in file_paths}
                        st.session_state.batch_results = {}
                        
                        st.success("🚀 Batch processing initialized! Click refresh to see interface.")
                        st.rerun()
                        
                    except Exception as e:
                        logger.error(f"Error initializing batch processing: {str(e)}")
                        st.error(f"Failed to initialize batch processing: {str(e)}")
            else:
                # Show processing interface
                show_batch_processing_status(components)
    
    else:
        st.info("👆 Upload PDF files to begin batch processing")


def process_single_file_batch(file_path: str, components: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single file in batch mode"""
    try:
        # Read file content
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        filename = Path(file_path).name
        
        # Validate file
        if not FileValidator.validate_pdf(file_content):
            return {"success": False, "error": "Invalid PDF file", "filename": filename}
        
        file_size_mb = FileValidator.get_file_size_mb(file_content)
        if not FileValidator.validate_file_size(file_content, max_size_mb=50):
            return {"success": False, "error": f"File too large ({file_size_mb:.1f}MB)", "filename": filename}
        
        start_time = time.time()
        
        # Step 1: Process PDF with LlamaParse
        doc_result = components["doc_processor"].process(file_path)
        if not doc_result["success"]:
            return {"success": False, "error": f"PDF processing failed: {doc_result.get('error', 'Unknown error')}", "filename": filename}
        
        documents = doc_result["documents"]
        
        # Step 2: Extract metadata
        full_text = "\n\n".join([doc.text for doc in documents])
        full_document = Document(text=full_text, metadata=documents[0].metadata)
        metadata_result = components["metadata_processor"].extract_metadata(full_document)
        metadata_dict = metadata_result.model_dump() if hasattr(metadata_result, 'model_dump') else metadata_result
        
        # Step 3: Chunk documents
        chunks = components["chunker"].chunk_document(full_document, metadata_dict)
        
        # Step 4: Index chunks
        index_result = components["indexer"].index_processed_chunks(chunks)
        if not index_result["success"]:
            return {"success": False, "error": f"Indexing failed: {index_result.get('error', 'Unknown error')}", "filename": filename}
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "filename": filename, 
            "processing_time": processing_time,
            "chunks_indexed": index_result.get("indexed_count", 0),
            "metadata": metadata_dict
        }
        
    except Exception as e:
        return {"success": False, "error": str(e), "filename": Path(file_path).name}


def show_batch_processing_status(components: Dict[str, Any]):
    """Show the batch processing status and controls"""
    st.subheader("🔄 Processing Status")
    
    # Auto-start processing if not started yet
    batch_status = st.session_state.get('batch_status', {})
    if batch_status and all(status == "PENDING" for status in batch_status.values()):
        # Check if this is the first time showing the interface
        if not st.session_state.get('batch_auto_started', False):
            st.session_state.batch_auto_started = True
            st.info("🚀 Auto-starting batch processing...")
            process_all_files_parallel(components)
            st.rerun()
    
    # Control buttons
    if st.button("⏹️ Stop Processing", type="secondary"):
        st.session_state.batch_processing_active = False
        st.success("Processing stopped")
        st.rerun()
    
    # Show configuration
    if st.session_state.get('batch_config'):
        config = st.session_state.batch_config
        file_paths = st.session_state.get('batch_file_paths', [])
        batch_status = st.session_state.get('batch_status', {})
        
        st.markdown("### 📋 Current Batch")
        
        # Calculate statistics
        total_files = len(file_paths)
        completed = sum(1 for status in batch_status.values() if status == "SUCCESS")
        failed = sum(1 for status in batch_status.values() if status == "ERROR")
        pending = sum(1 for status in batch_status.values() if status == "PENDING")
        processing = sum(1 for status in batch_status.values() if status == "PROCESSING")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Files", total_files)
        with col2:
            st.metric("Completed", completed, delta=f"{completed/total_files*100:.1f}%" if total_files > 0 else "0%")
        with col3:
            st.metric("Failed", failed)
        with col4:
            st.metric("Pending", pending)
        
        # Progress bar
        if total_files > 0:
            progress = (completed + failed) / total_files
            st.progress(progress, text=f"Progress: {completed + failed}/{total_files} files processed")
        
        # Show file list with real status
        st.markdown("### 📄 Document Status")
        for file_path in file_paths:
            filename = Path(file_path).name
            status = batch_status.get(file_path, "PENDING")
            
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"**{filename}**")
                with col2:
                    if status == "SUCCESS":
                        st.success(status)
                    elif status == "ERROR":
                        st.error(status)
                    elif status == "PROCESSING":
                        st.warning(status)
                    else:
                        st.info(status)
                with col3:
                    # Show result details if available
                    results = st.session_state.get('batch_results', {})
                    if file_path in results:
                        result = results[file_path]
                        if result.get('success'):
                            st.caption(f"{result.get('chunks_indexed', 0)} chunks")
                        else:
                            st.caption("Error")
        
        # Show detailed results if any
        results = st.session_state.get('batch_results', {})
        if results:
            successful_results = [r for r in results.values() if r.get('success')]
            if successful_results:
                total_chunks = sum(r.get('chunks_indexed', 0) for r in successful_results)
                avg_time = sum(r.get('processing_time', 0) for r in successful_results) / len(successful_results)
                st.info(f"Successfully indexed {total_chunks} chunks total (avg time: {avg_time:.1f}s per document)")
        
        # Clear processing button
        if st.button("🗑️ Clear Session", type="secondary"):
            # Clean up temporary files
            import shutil
            temp_dir = st.session_state.get('batch_temp_dir')
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            # Clear session state
            for key in list(st.session_state.keys()):
                if key.startswith('batch_'):
                    del st.session_state[key]
            
            st.success("Session cleared")
            st.rerun()



def process_all_files_parallel(components: Dict[str, Any]):
    """Process all pending files using parallel workflow manager"""
    import asyncio
    
    batch_status = st.session_state.get('batch_status', {})
    file_paths = st.session_state.get('batch_file_paths', [])
    config = st.session_state.get('batch_config')
    
    pending_files = [fp for fp in file_paths if batch_status.get(fp) == "PENDING"]
    
    if not pending_files:
        st.info("No pending files to process")
        return
    
    # Progress tracking variables
    progress_bar = st.progress(0, text="Starting parallel batch processing...")
    progress_container = st.empty()
    
    def progress_callback(progress_event):
        """Update Streamlit interface based on progress events"""
        try:
            # Update individual file status
            file_path = getattr(progress_event, 'file_path', None)
            status = getattr(progress_event, 'status', None)
            
            if file_path and status:
                if hasattr(progress_event, 'processing_status'):
                    if progress_event.processing_status == ProcessingStatus.PROCESSING:
                        st.session_state.batch_status[file_path] = "PROCESSING"
                    elif progress_event.processing_status == ProcessingStatus.COMPLETED:
                        st.session_state.batch_status[file_path] = "SUCCESS"
                    elif progress_event.processing_status == ProcessingStatus.FAILED:
                        st.session_state.batch_status[file_path] = "ERROR"
            
            # Update overall progress
            if hasattr(progress_event, 'progress'):
                progress = progress_event.progress
                progress_bar.progress(progress, text=f"Processing files... ({progress:.0%})")
                
        except Exception as e:
            logger.warning(f"Error in progress callback: {e}")
    
    try:
        st.info(f"🚀 Starting parallel processing of {len(pending_files)} files with {config.max_parallel_workflows} workers")
        
        # Run async function in event loop
        async def run_parallel_processing():
            return await process_documents_batch(
                pending_files,
                config=config,
                progress_callback=progress_callback
            )
        
        # Execute the async function
        result = asyncio.run(run_parallel_processing())
        
        # Update session state with results
        if 'batch_results' not in st.session_state:
            st.session_state.batch_results = {}
        
        # Process batch result
        successful = result.successful_count
        failed = result.failed_count
        
        # Update individual file results - process successful results
        for success_result in result.successful_results:
            file_path = success_result.file_path
            st.session_state.batch_status[file_path] = "SUCCESS" 
            st.session_state.batch_results[file_path] = {
                'success': True,
                'filename': success_result.filename,
                'processing_time': getattr(success_result, 'indexing_time', 0),
                'chunks_indexed': success_result.chunks_indexed
            }
        
        # Process failed results
        for error_result in result.failed_results:
            file_path = error_result.file_path
            st.session_state.batch_status[file_path] = "ERROR"
            st.session_state.batch_results[file_path] = {
                'success': False,
                'filename': error_result.filename,
                'error': error_result.error_message or "Unknown error"
            }
        
        # Complete progress
        progress_bar.progress(1.0, text="Parallel batch processing completed!")
        
        # Show summary
        if failed == 0:
            st.success(f"All {successful} files processed successfully in parallel!")
        else:
            st.warning(f"Parallel batch completed: {successful} successful, {failed} failed")
            
    except Exception as e:
        logger.error(f"Error in parallel processing: {e}")
        st.error(f"Parallel processing failed: {str(e)}")
        
        # Mark all pending files as failed
        for file_path in pending_files:
            if st.session_state.batch_status.get(file_path) == "PENDING":
                st.session_state.batch_status[file_path] = "ERROR"


def main():
    """Main application function"""
    # App title and description
    st.title("📚 ArXiv RAG System")
    st.markdown("""
    **Система анализа научных публикаций ArXiv** с фокусом на экономику, квантовые финансы и машинное обучение.
    
    Загрузите PDF документы, задавайте вопросы и получайте научно обоснованные ответы на русском языке.
    """)
    
    # Initialize components
    components = initialize_components()
    
    # Sidebar navigation
    with st.sidebar:
        st.header("📋 Navigation")
        tab = st.radio(
            "Choose section:",
            ["🔍 Search", "📤 Upload Document", "🔄 Batch Processing", "📝 Query History", "📊 Statistics"]
        )
        
        # API Key status
        st.header("🔑 API Keys Status")
        api_validation = ConfigValidator.validate_api_keys()
        for key_name, is_valid in api_validation.items():
            status = "✅" if is_valid else "❌"
            st.write(f"{status} {key_name.upper()}")
        
        # Quick stats
        st.header("📈 Quick Stats")
        cache_stats = components["query_cache"].get_stats()
        st.metric("Cached Queries", cache_stats.get("active_items", 0))
        
        system_metrics = metrics.get_stats()
        total_queries = system_metrics.get("counters", {}).get("queries_processed", 0)
        st.metric("Total Queries", total_queries)
    
    # Handle selected query from history
    if hasattr(st.session_state, 'selected_query'):
        st.info(f"Rerunning query: {st.session_state.selected_query}")
        # Clear the selected query
        del st.session_state.selected_query
    
    # Main content based on selected tab
    if tab == "🔍 Search":
        create_search_interface(components)
    
    elif tab == "📤 Upload Document":
        st.header("📤 Upload PDF Document")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload an ArXiv PDF document for analysis. Maximum size: 50MB"
        )
        
        if uploaded_file is not None:
            # Display file info
            file_size_mb = FileValidator.get_file_size_mb(uploaded_file.getvalue())
            st.info(f"File: {uploaded_file.name} ({file_size_mb:.1f}MB)")
            
            if st.button("📤 Process Document", type="primary"):
                result = process_uploaded_pdf(uploaded_file, components)
                
                if result and result["success"]:
                    st.success(f"✅ Document processed successfully in {format_processing_time(result['processing_time'])}")
                    
                    # Display processing results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Chunks Created", result["chunks_created"])
                    with col2:
                        st.metric("Chunks Indexed", result["chunks_indexed"])
                    with col3:
                        st.metric("Processing Time", format_processing_time(result["processing_time"]))
                    
                    # Display metadata
                    if result.get("metadata"):
                        display_document_metadata(result["metadata"])
    
    elif tab == "🔄 Batch Processing":
        create_batch_processing_interface(components)
    
    elif tab == "📝 Query History":
        create_history_interface(components)
    
    elif tab == "📊 Statistics":
        create_statistics_interface(components)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>ArXiv RAG System | Economics, Quantum Finance, and Machine Learning Analysis</p>
        <p><small>Powered by LlamaIndex, Pinecone, OpenAI, and Cohere</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()