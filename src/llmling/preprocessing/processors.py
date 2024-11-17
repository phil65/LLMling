"""Concrete preprocessor implementations."""

import re
from typing import Any

from bs4 import BeautifulSoup

try:
    import nltk
except ImportError:
    nltk = None

try:
    from markdown.preprocessors import Preprocessor as MarkdownPreprocessor
except ImportError:
    MarkdownPreprocessor = None

from .base import BasePreprocessor, ProcessingContext, ProcessingResult


class TextNormalizer(BasePreprocessor):
    """Normalizes text content."""
    
    @property
    def name(self) -> str:
        return "text_normalizer"
    
    def _process_impl(self, context: ProcessingContext) -> ProcessingResult:
        content = context.original_content
        
        # Normalize whitespace
        content = " ".join(content.split())
        
        # Normalize quotes
        content = re.sub(r'["""]', '"', content)
        content = re.sub(r'[''']', "'", content)
        
        # Normalize dashes
        content = re.sub(r'[‒–—―]', '-', content)
        
        return ProcessingResult(
            content=content,
            stats={
                "chars_before": len(context.original_content),
                "chars_after": len(content)
            }
        )


class HTMLCleaner(BasePreprocessor):
    """Cleans HTML content."""
    
    @property
    def name(self) -> str:
        return "html_cleaner"
    
    def _process_impl(self, context: ProcessingContext) -> ProcessingResult:
        if not context.content_type.startswith("text/html"):
            return ProcessingResult(content=context.original_content)
            
        soup = BeautifulSoup(context.original_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text content
        content = soup.get_text()
        
        # Clean up whitespace
        content = " ".join(content.split())
        
        return ProcessingResult(
            content=content,
            metadata={"extracted_tags": list(set(tag.name for tag in soup.find_all()))}
        )


class SentenceSegmenter(BasePreprocessor):
    """Segments text into sentences."""
    
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        if nltk is not None:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
    
    @property
    def name(self) -> str:
        return "sentence_segmenter"
    
    def _process_impl(self, context: ProcessingContext) -> ProcessingResult:
        if nltk is None:
            return ProcessingResult(
                content=context.original_content,
                metadata={"error": "NLTK not installed"}
            )
            
        sentences = nltk.sent_tokenize(context.original_content)
        content = "\n".join(sentences)
        
        return ProcessingResult(
            content=content,
            stats={
                "sentence_count": len(sentences),
                "avg_sentence_length": len(content) / len(sentences)
            }
        )


class RedundancyRemover(BasePreprocessor):
    """Removes redundant content."""
    
    @property
    def name(self) -> str:
        return "redundancy_remover"
    
    def _process_impl(self, context: ProcessingContext) -> ProcessingResult:
        lines = context.original_content.split("\n")
        unique_lines = []
        seen = set()
        
        for line in lines:
            line_normalized = line.strip().lower()
            if line_normalized and line_normalized not in seen:
                seen.add(line_normalized)
                unique_lines.append(line)
        
        content = "\n".join(unique_lines)
        
        return ProcessingResult(
            content=content,
            stats={
                "lines_before": len(lines),
                "lines_after": len(unique_lines)
            }
        )


class CodeBlockExtractor(BasePreprocessor):
    """Extracts and processes code blocks."""
    
    @property
    def name(self) -> str:
        return "code_block_extractor"
    
    def _process_impl(self, context: ProcessingContext) -> ProcessingResult:
        content = context.original_content
        code_blocks = []
        
        # Extract markdown code blocks
        pattern = r'```[\w]*\n(.*?)\n```'
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            code_blocks.append(match.group(1))
            
        if code_blocks:
            metadata = {
                "code_block_count": len(code_blocks),
                "code_blocks": code_blocks
            }
        else:
            metadata = {"code_block_count": 0}
            
        return ProcessingResult(
            content=content,
            metadata=metadata
        )


class MarkdownPreprocessorAdapter(BasePreprocessor):
    """Adapter for markdown.preprocessors.Preprocessor."""
    
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        if MarkdownPreprocessor is None:
            raise ImportError(
                "Python-Markdown package not installed. "
                "Install with: pip install markdown"
            )
        
        self.markdown_preprocessor = self._create_preprocessor()
    
    @property
    def name(self) -> str:
        return "markdown_preprocessor"
    
    def _create_preprocessor(self) -> MarkdownPreprocessor:
        """Create the markdown preprocessor instance."""
        class CustomMarkdownPreprocessor(MarkdownPreprocessor):
            def run(self, lines: list[str]) -> list[str]:
                # Apply any custom preprocessing rules here
                processed = []
                for line in lines:
                    # Example: Remove HTML comments
                    line = re.sub(r'<!--.*?-->', '', line)
                    processed.append(line)
                return processed
        
        return CustomMarkdownPreprocessor()
    
    def _process_impl(self, context: ProcessingContext) -> ProcessingResult:
        # Split content into lines as required by markdown preprocessor
        lines = context.original_content.split('\n')
        
        # Run markdown preprocessing
        processed_lines = self.markdown_preprocessor.run(lines)
        
        # Join lines back into content
        content = '\n'.join(processed_lines)
        
        return ProcessingResult(
            content=content,
            stats={
                "lines_processed": len(lines)
            },
            metadata={
                "preprocessor_type": self.markdown_preprocessor.__class__.__name__
            }
        )
