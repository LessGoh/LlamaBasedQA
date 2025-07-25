"""
Hybrid chunking module for different content types

This module provides hybrid chunking strategies for different content types:
- Tables: as separate chunks
- Formulas: with surrounding context  
- Regular text: fixed size with overlap
Based on PRD specifications for ArXiv scientific document analysis.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from .config import settings, get_chunk_metadata_template

# Set up logger
logger = logging.getLogger(__name__)


class ChunkingConfig(BaseModel):
    """Configuration for hybrid chunking"""
    
    chunk_size: int = Field(default=settings.max_chunk_size)  # 1024 tokens
    chunk_overlap: float = Field(default=settings.chunk_overlap)  # 0.15 (15%)
    min_chunk_size: int = Field(default=50)  # Minimum chunk size
    preserve_sentences: bool = Field(default=True)  # Preserve sentence boundaries
    
    # Content type specific settings
    table_chunk_separately: bool = Field(default=True)
    formula_context_size: int = Field(default=200)  # Characters before/after formula
    figure_caption_chunk_separately: bool = Field(default=True)
    
    # Detection patterns
    table_patterns: List[str] = Field(default_factory=lambda: [
        r'\|.*\|.*\|',  # Markdown table pattern
        r'^\s*\|[^|]*\|[^|]*\|',  # Table row pattern
        r'Table\s+\d+',  # Table reference
        r'\\begin\{table\}',  # LaTeX table
        r'\\begin\{tabular\}'  # LaTeX tabular
    ])
    
    formula_patterns: List[str] = Field(default_factory=lambda: [
        r'\$[^$]+\$',  # Inline LaTeX
        r'\$\$[^$]+\$\$',  # Display LaTeX
        r'\\begin\{equation\}',  # LaTeX equation
        r'\\begin\{align\}',  # LaTeX align
        r'\\begin\{gather\}',  # LaTeX gather
        r'\\[\[\]][^\\]*\\[\]\]]'  # Alternative LaTeX delimiters
    ])
    
    figure_patterns: List[str] = Field(default_factory=lambda: [
        r'Figure\s+\d+',  # Figure reference
        r'Fig\.\s*\d+',  # Fig. reference  
        r'\\begin\{figure\}',  # LaTeX figure
        r'!\[.*\]\(.*\)'  # Markdown image
    ])


class ChunkMetadata(BaseModel):
    """Metadata for a chunk"""
    
    page_number: int = Field(default=0)
    
    # Additional metadata
    original_text_length: int = Field(default=0)
    
    # Context information
    previous_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    parent_section: Optional[str] = None


# ЗАКОММЕНТИРОВАНО: ContentDetector больше не используется в упрощенном чанкинге
# class ContentDetector:
#     """
#     Content type detector for identifying tables, formulas, and figures
#     """
    
    def __init__(self, config: ChunkingConfig):
        """
        Initialize content detector
        
        Args:
            config: Chunking configuration
        """
        self.config = config
        self._compile_patterns()
        logger.info("ContentDetector initialized")
    
    def _compile_patterns(self):
        """Compile regex patterns for better performance"""
        self.table_regexes = [re.compile(pattern, re.MULTILINE | re.IGNORECASE) 
                             for pattern in self.config.table_patterns]
        self.formula_regexes = [re.compile(pattern, re.MULTILINE | re.DOTALL)
                               for pattern in self.config.formula_patterns]
        self.figure_regexes = [re.compile(pattern, re.MULTILINE | re.IGNORECASE)
                              for pattern in self.config.figure_patterns]
    
    def detect_content_type(self, text: str) -> str:
        """
        Detect the primary content type of a text segment
        
        Args:
            text: Text to analyze
            
        Returns:
            Content type: 'table', 'formula', 'figure_caption', or 'text'
        """
        text_strip = text.strip()
        
        # Check for tables
        if self.is_table(text_strip):
            return "table"
        
        # Check for formulas
        if self.is_formula(text_strip):
            return "formula"
        
        # Check for figure captions
        if self.is_figure_caption(text_strip):
            return "figure_caption"
        
        return "text"
    
    def is_table(self, text: str) -> bool:
        """
        Check if text contains a table
        
        Args:
            text: Text to check
            
        Returns:
            True if text contains table content
        """
        # Check for multiple pipe characters (markdown table)
        pipe_count = text.count('|')
        if pipe_count >= 6:  # At least 3 columns with headers and one row
            return True
        
        # Check regex patterns
        for regex in self.table_regexes:
            if regex.search(text):
                return True
        
        # Check for structured data patterns
        lines = text.split('\n')
        tabular_lines = 0
        for line in lines:
            if '|' in line and line.count('|') >= 2:
                tabular_lines += 1
        
        # If more than 2 lines look tabular, it's likely a table
        return tabular_lines >= 2
    
    def is_formula(self, text: str) -> bool:
        """
        Check if text contains mathematical formulas
        
        Args:
            text: Text to check
            
        Returns:
            True if text contains formulas
        """
        # Check for LaTeX math delimiters
        if '$' in text and text.count('$') >= 2:
            return True
        
        # Check regex patterns
        for regex in self.formula_regexes:
            if regex.search(text):
                return True
        
        # Check for common mathematical symbols
        math_symbols = ['∑', '∫', '∏', '√', '∞', '≤', '≥', '≠', '±', '÷', '×', 'α', 'β', 'γ', 'λ', 'μ', 'σ', 'π']
        math_count = sum(1 for symbol in math_symbols if symbol in text)
        
        # If text has multiple math symbols, likely a formula
        return math_count >= 3
    
    def is_figure_caption(self, text: str) -> bool:
        """
        Check if text is a figure caption
        
        Args:
            text: Text to check
            
        Returns:
            True if text is a figure caption
        """
        # Check regex patterns
        for regex in self.figure_regexes:
            if regex.search(text):
                return True
        
        # Check for caption-like patterns
        lower_text = text.lower().strip()
        if (lower_text.startswith('figure') or 
            lower_text.startswith('fig.') or
            lower_text.startswith('image') or
            'caption:' in lower_text):
            return True
        
        return False
    
    def count_formulas(self, text: str) -> int:
        """
        Count number of formulas in text
        
        Args:
            text: Text to analyze
            
        Returns:
            Number of formulas found
        """
        count = 0
        
        # Count LaTeX math expressions
        count += len(re.findall(r'\$[^$]+\$', text))  # Inline math
        count += len(re.findall(r'\$\$[^$]+\$\$', text))  # Display math
        
        # Count LaTeX environments
        for pattern in ['equation', 'align', 'gather', 'eqnarray']:
            count += len(re.findall(rf'\\begin\{{{pattern}\}}', text, re.IGNORECASE))
        
        return count
    
    def count_tables(self, text: str) -> int:
        """
        Count number of tables in text
        
        Args:
            text: Text to analyze
            
        Returns:
            Number of tables found
        """
        count = 0
        
        # Count table references
        count += len(re.findall(r'Table\s+\d+', text, re.IGNORECASE))
        
        # Count LaTeX table environments
        count += len(re.findall(r'\\begin\{table\}', text, re.IGNORECASE))
        count += len(re.findall(r'\\begin\{tabular\}', text, re.IGNORECASE))
        
        # Count markdown tables (rough estimate)
        lines = text.split('\n')
        table_blocks = 0
        in_table = False
        
        for line in lines:
            if '|' in line and line.count('|') >= 2:
                if not in_table:
                    table_blocks += 1
                    in_table = True
            else:
                in_table = False
        
        count += table_blocks
        
        return count


class SectionExtractor:
    """
    Extract sections and hierarchical structure from document
    """
    
    def __init__(self):
        """Initialize section extractor"""
        # Patterns for different heading levels
        self.heading_patterns = [
            r'^#{1,6}\s+(.+)$',  # Markdown headings
            r'^([A-Z][A-Z\s]+)$',  # ALL CAPS headings
            r'^\d+\.?\s+([A-Z][^.]+)$',  # Numbered sections
            r'^([A-Z][a-z]+(\s+[A-Z][a-z]+)*):?\s*$',  # Title case headings
            r'\\section\{([^}]+)\}',  # LaTeX sections
            r'\\subsection\{([^}]+)\}',  # LaTeX subsections
        ]
        self.compiled_patterns = [re.compile(pattern, re.MULTILINE) 
                                for pattern in self.heading_patterns]
        logger.info("SectionExtractor initialized")
    
    def extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract sections with titles from document
        
        Args:
            text: Document text
            
        Returns:
            List of section dictionaries with metadata
        """
        sections = []
        lines = text.split('\n')
        current_section = {
            "title": "Introduction",
            "content": "",
            "start_line": 0,
            "page": 1,
            "level": 1
        }
        
        line_num = 0
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check if line is a heading
            heading_info = self._detect_heading(line_stripped)
            if heading_info:
                # Save current section if it has content
                if current_section["content"].strip():
                    current_section["end_line"] = i - 1
                    sections.append(current_section.copy())
                
                # Start new section
                current_section = {
                    "title": heading_info["title"],
                    "content": "",
                    "start_line": i,
                    "page": self._estimate_page(i, len(lines)),
                    "level": heading_info["level"]
                }
            else:
                # Add line to current section content
                current_section["content"] += line + "\n"
        
        # Add final section
        if current_section["content"].strip():
            current_section["end_line"] = len(lines) - 1
            sections.append(current_section)
        
        logger.info(f"Extracted {len(sections)} sections")
        return sections
    
    def _detect_heading(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Detect if line is a heading and extract info
        
        Args:
            line: Line to check
            
        Returns:
            Heading info dictionary or None
        """
        # Markdown headings
        markdown_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if markdown_match:
            level = len(markdown_match.group(1))
            title = markdown_match.group(2).strip()
            return {"title": title, "level": level, "type": "markdown"}
        
        # ALL CAPS headings (likely major sections)
        if re.match(r'^[A-Z][A-Z\s]{3,}$', line) and len(line.strip()) < 50:
            return {"title": line.strip(), "level": 1, "type": "caps"}
        
        # Numbered sections
        numbered_match = re.match(r'^(\d+\.?\d*\.?)\s+([A-Z][^.]+)$', line)
        if numbered_match:
            level = numbered_match.group(1).count('.') + 1
            title = numbered_match.group(2).strip()
            return {"title": title, "level": level, "type": "numbered"}
        
        # LaTeX sections
        latex_match = re.search(r'\\(sub)*section\{([^}]+)\}', line)
        if latex_match:
            level = 1 if latex_match.group(1) is None else 2
            title = latex_match.group(2).strip()
            return {"title": title, "level": level, "type": "latex"}
        
        return None
    
    def _estimate_page(self, line_num: int, total_lines: int, lines_per_page: int = 50) -> int:
        """
        Estimate page number based on line position
        
        Args:
            line_num: Current line number
            total_lines: Total number of lines
            lines_per_page: Estimated lines per page
            
        Returns:
            Estimated page number
        """
        return max(1, (line_num // lines_per_page) + 1)


class HybridChunker:
    """
    Hybrid chunking strategy for different content types
    
    This class implements the hybrid chunking approach specified in PRD:
    - Tables: as separate chunks
    - Formulas: with surrounding context
    - Regular text: fixed size with 10-20% overlap
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize simplified chunker
        
        Args:
            config: Optional configuration override
        """
        self.config = config or ChunkingConfig()
        # Убрать: self.detector = ContentDetector(self.config)
        # Убрать: self.section_extractor = SectionExtractor()
        
        # Оставить только SentenceSplitter
        self.sentence_splitter = SentenceSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=int(self.config.chunk_size * self.config.chunk_overlap),
            separator=" "
        )
        
        logger.info(f"HybridChunker initialized with simplified chunking, chunk_size: {self.config.chunk_size}")
    
    def chunk_document(self, document: Document, metadata: Optional[Dict[str, Any]] = None) -> List[TextNode]:
        """
        Apply simplified chunking strategy to document
        
        Args:
            document: Document to chunk
            metadata: Optional document-level metadata
            
        Returns:
            List of TextNode objects with appropriate metadata
        """
        try:
            logger.info("Starting simplified chunking")
            
            # Initialize metadata (БЕЗ ИЗМЕНЕНИЙ)
            doc_metadata = metadata or {}
            if hasattr(document, 'metadata') and document.metadata:
                doc_metadata.update(document.metadata)
            
            # УПРОЩЕНИЕ: Прямое разбиение всего текста на чанки
            text_chunks = self.sentence_splitter.split_text(document.text)
            
            chunks = []
            for i, chunk_text in enumerate(text_chunks):
                # Сохранить функцию создания TextNode с полными метаданными
                chunk_node = self._create_text_node(
                    text=chunk_text,
                    chunk_type="text",  # Всегда текст
                    section_title="Document",  # Простое название
                    page_number=self._estimate_page_number(chunk_text, document.text),
                    chunk_index=i,
                    doc_metadata=doc_metadata
                )
                chunks.append(chunk_node)
            
            # Сохранить все функции метаданных
            self._add_chunk_relationships(chunks)
            self.log_chunk_structure(chunks)
            stats = self.get_chunking_stats(chunks)
            
            logger.info(f"Simplified chunking completed: {len(chunks)} chunks created")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in simplified chunking: {str(e)}")
            raise
    
    def _estimate_page_number(self, chunk_text: str, full_text: str) -> int:
        """
        Estimate page number based on chunk position in document
        
        Args:
            chunk_text: Text of the chunk
            full_text: Full document text
            
        Returns:
            Estimated page number
        """
        chunk_position = full_text.find(chunk_text)
        if chunk_position == -1:
            return 1
        
        # Примерно 2000 символов на страницу
        chars_per_page = 2000
        return max(1, (chunk_position // chars_per_page) + 1)
    
    def _process_section(self, section: Dict[str, Any], doc_metadata: Dict[str, Any], 
                        start_index: int) -> List[TextNode]:
        """
        Process a single section with hybrid chunking
        
        Args:
            section: Section dictionary
            doc_metadata: Document-level metadata
            start_index: Starting chunk index
            
        Returns:
            List of TextNode objects for the section
        """
        section_content = section["content"]
        section_title = section["title"]
        page_number = section["page"]
        
        logger.info(f"Processing section: \"{section_title}\" (page {page_number}, {len(section_content)} chars)")
        
        # Detect special content types in the section
        special_chunks = self._extract_special_content(section_content)
        if special_chunks:
            logger.info(f"Found {len(special_chunks)} special content chunks in section \"{section_title}\"")
            for i, special in enumerate(special_chunks):
                logger.info(f"  Special chunk {i}: type={special['type']}, size={len(special['text'])} chars")
        
        chunks = []
        chunk_index = start_index
        
        if special_chunks:
            # Process section with special content
            processed_text = section_content
            
            for special_chunk in special_chunks:
                # Create chunk for special content
                special_node = self._create_text_node(
                    text=special_chunk["text"],
                    chunk_type=special_chunk["type"],
                    section_title=section_title,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    doc_metadata=doc_metadata,
                    additional_metadata=special_chunk.get("metadata", {})
                )
                chunks.append(special_node)
                chunk_index += 1
                
                # Remove special content from processed text for regular chunking
                processed_text = processed_text.replace(special_chunk["text"], "", 1)
            
            # Process remaining text normally if significant content remains
            if len(processed_text.strip()) > self.config.min_chunk_size:
                regular_chunks = self._chunk_regular_text(
                    processed_text, section_title, page_number, 
                    chunk_index, doc_metadata
                )
                chunks.extend(regular_chunks)
        else:
            # Process as regular text
            regular_chunks = self._chunk_regular_text(
                section_content, section_title, page_number,
                chunk_index, doc_metadata
            )
            chunks.extend(regular_chunks)
        
        logger.info(f"Section \"{section_title}\" processed -> created {len(chunks)} chunks")
        
        return chunks
    
    def _extract_special_content(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract tables, formulas, and figure captions from text
        
        Args:
            text: Text to process
            
        Returns:
            List of special content chunks
        """
        special_chunks = []
        
        # Extract tables
        if self.config.table_chunk_separately:
            tables = self._extract_tables(text)
            for table in tables:
                special_chunks.append({
                    "text": table["text"],
                    "type": "table",
                    "metadata": {
                        "table_rows": table.get("rows", 0),
                        "table_columns": table.get("columns", 0)
                    }
                })
        
        # Extract formulas with context
        formulas = self._extract_formulas_with_context(text)
        for formula in formulas:
            special_chunks.append({
                "text": formula["text"],
                "type": "formula",
                "metadata": {
                    "formula_type": formula.get("formula_type", "unknown"),
                    "has_context": formula.get("has_context", False)
                }
            })
        
        # Extract figure captions
        if self.config.figure_caption_chunk_separately:
            captions = self._extract_figure_captions(text)
            for caption in captions:
                special_chunks.append({
                    "text": caption["text"],
                    "type": "figure_caption",
                    "metadata": {
                        "figure_number": caption.get("figure_number", "unknown")
                    }
                })
        
        return special_chunks
    
    def _extract_tables(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract table content from text
        
        Args:
            text: Text to process
            
        Returns:
            List of table dictionaries
        """
        tables = []
        lines = text.split('\n')
        
        current_table = None
        table_lines = []
        
        for line in lines:
            if self.detector.is_table(line):
                if current_table is None:
                    # Start new table
                    current_table = {"start": len(tables)}
                table_lines.append(line)
            else:
                if current_table is not None and table_lines:
                    # End current table
                    table_text = '\n'.join(table_lines)
                    rows = len(table_lines)
                    columns = max((line.count('|') - 1 for line in table_lines if '|' in line), default=0)
                    
                    tables.append({
                        "text": table_text,
                        "rows": rows,
                        "columns": max(0, columns)
                    })
                    
                    current_table = None
                    table_lines = []
        
        # Handle table at end of text
        if current_table is not None and table_lines:
            table_text = '\n'.join(table_lines)
            rows = len(table_lines)
            columns = max((line.count('|') - 1 for line in table_lines if '|' in line), default=0)
            
            tables.append({
                "text": table_text,
                "rows": rows,
                "columns": max(0, columns)
            })
        
        return tables
    
    def _extract_formulas_with_context(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract formulas with surrounding context
        
        Args:
            text: Text to process
            
        Returns:
            List of formula dictionaries with context
        """
        formulas = []
        
        # Find LaTeX math expressions
        patterns = [
            (r'\$\$([^$]+)\$\$', 'display'),  # Display math
            (r'\$([^$]+)\$', 'inline'),       # Inline math
            (r'\\begin\{equation\}(.*?)\\end\{equation\}', 'equation'),  # Equations
            (r'\\begin\{align\}(.*?)\\end\{align\}', 'align'),          # Align
        ]
        
        for pattern, formula_type in patterns:
            for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
                start_pos = match.start()
                end_pos = match.end()
                formula_text = match.group(0)
                
                # Add context around formula
                context_start = max(0, start_pos - self.config.formula_context_size)
                context_end = min(len(text), end_pos + self.config.formula_context_size)
                
                context_text = text[context_start:context_end]
                
                formulas.append({
                    "text": context_text,
                    "formula_type": formula_type,
                    "has_context": context_start < start_pos or context_end > end_pos,
                    "pure_formula": formula_text
                })
        
        return formulas
    
    def _extract_figure_captions(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract figure captions from text
        
        Args:
            text: Text to process
            
        Returns:
            List of figure caption dictionaries
        """
        captions = []
        lines = text.split('\n')
        
        for line in lines:
            if self.detector.is_figure_caption(line.strip()):
                # Extract figure number if present
                figure_match = re.search(r'Figure\s+(\d+)', line, re.IGNORECASE)
                figure_number = figure_match.group(1) if figure_match else "unknown"
                
                captions.append({
                    "text": line.strip(),
                    "figure_number": figure_number
                })
        
        return captions
    
    def _chunk_regular_text(self, text: str, section_title: str, page_number: int,
                           start_index: int, doc_metadata: Dict[str, Any]) -> List[TextNode]:
        """
        Chunk regular text using sentence splitter
        
        Args:
            text: Text to chunk
            section_title: Section title
            page_number: Page number
            start_index: Starting chunk index
            doc_metadata: Document metadata
            
        Returns:
            List of TextNode objects
        """
        if not text.strip() or len(text.strip()) < self.config.min_chunk_size:
            return []
        
        # Split text into chunks
        text_chunks = self.sentence_splitter.split_text(text)
        
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_node = self._create_text_node(
                text=chunk_text,
                chunk_type="text",
                section_title=section_title,
                page_number=page_number,
                chunk_index=start_index + i,
                doc_metadata=doc_metadata
            )
            chunks.append(chunk_node)
        
        return chunks
    
    def _create_text_node(self, text: str, chunk_type: str, section_title: str,
                         page_number: int, chunk_index: int, doc_metadata: Dict[str, Any],
                         additional_metadata: Optional[Dict[str, Any]] = None) -> TextNode:
        """
        Create TextNode with proper metadata
        
        Args:
            text: Chunk text
            chunk_type: Type of chunk (kept for backward compatibility)
            section_title: Section title (kept for backward compatibility)
            page_number: Page number
            chunk_index: Chunk index (kept for backward compatibility)
            doc_metadata: Document metadata
            additional_metadata: Additional chunk-specific metadata
            
        Returns:
            TextNode with metadata
        """
        # Create chunk metadata
        chunk_metadata = ChunkMetadata(
            page_number=page_number,
            original_text_length=len(text)
        )
        
        # Combine all metadata
        full_metadata = {
            **doc_metadata,  # Document-level metadata
            **chunk_metadata.model_dump(),  # Chunk-level metadata
            **(additional_metadata or {})  # Additional metadata
        }
        
        # Generate chunk ID
        chunk_id = f"chunk_{hash(text[:50])}"
        
        # Create TextNode
        node = TextNode(
            text=text,
            metadata=full_metadata,
            id_=chunk_id,
        )
        
        # Log chunk creation with ID details
        logger.info(
            f"Created TextNode: id={chunk_id}, "
            f"size={len(text)} chars, page={page_number}"
        )
        
        return node
    
    def _add_chunk_relationships(self, chunks: List[TextNode]):
        """
        Add relationships between chunks (previous/next)
        
        Args:
            chunks: List of chunks to process
        """
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.metadata["previous_chunk_id"] = chunks[i-1].id_
            if i < len(chunks) - 1:
                chunk.metadata["next_chunk_id"] = chunks[i+1].id_
    
    def get_chunking_stats(self, chunks: List[TextNode]) -> Dict[str, Any]:
        """
        Get statistics about chunking results
        
        Args:
            chunks: List of chunks
            
        Returns:
            Dictionary with chunking statistics
        """
        if not chunks:
            return {"total_chunks": 0}
        
        total_chars = 0
        
        for chunk in chunks:
            total_chars += len(chunk.text)
        
        stats = {
            "total_chunks": len(chunks),
            "total_characters": total_chars,
            "average_chars_per_chunk": total_chars / len(chunks),
            "chunk_size_config": self.config.chunk_size,
            "chunk_overlap_config": self.config.chunk_overlap
        }
        
        logger.info(f"Chunking stats: {stats}")
        return stats
    
    def log_chunk_structure(self, chunks: List[TextNode]):
        """
        Log detailed structure of created chunks for debugging and monitoring
        
        Args:
            chunks: List of created chunks
        """
        if not chunks:
            logger.info("No chunks to display")
            return
        
        logger.info(f"=== Chunk Structure Details ===")
        
        # Log each chunk with detailed information
        for i, chunk in enumerate(chunks):
            text_length = len(chunk.text)
            page_number = chunk.metadata.get("page_number", 0)
            
            # Create preview (first 100 characters, clean up whitespace)
            preview_text = chunk.text.strip().replace('\n', ' ').replace('\r', '')
            preview = preview_text[:100] + "..." if len(preview_text) > 100 else preview_text
            
            logger.info(
                f"Chunk {i}: page={page_number}, "
                f"size={text_length} chars, "
                f"preview=\"{preview}\""
            )
        
        logger.info(f"Chunking summary: {len(chunks)} total chunks")
        logger.info(f"=== End Chunk Structure Details ===")


# Convenience functions for easy usage
def chunk_document(document: Document, **kwargs) -> List[TextNode]:
    """
    Convenience function to chunk document with default settings
    
    Args:
        document: Document to chunk
        **kwargs: Additional configuration options
        
    Returns:
        List of TextNode objects
    """
    config = ChunkingConfig(**kwargs)
    chunker = HybridChunker(config)
    return chunker.chunk_document(document)


def chunk_text(text: str, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> List[TextNode]:
    """
    Convenience function to chunk raw text
    
    Args:
        text: Text to chunk
        metadata: Optional metadata
        **kwargs: Additional configuration options
        
    Returns:
        List of TextNode objects
    """
    document = Document(text=text, metadata=metadata or {})
    return chunk_document(document, **kwargs)