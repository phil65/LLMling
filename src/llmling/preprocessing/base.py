"""Base classes and interfaces for context preprocessing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel


class ProcessingError(Exception):
    """Base exception for preprocessing errors."""
    pass


class ProcessingResult(BaseModel):
    """Result of a preprocessing operation."""
    content: str
    metadata: dict[str, Any] = {}
    stats: dict[str, Any] = {}


class ProcessingContext(BaseModel):
    """Context for preprocessing operations."""
    content_id: UUID
    content_type: str
    original_content: str
    metadata: dict[str, Any] = {}


class IPreprocessor(Protocol):
    """Interface for preprocessor implementations."""
    
    def process(self, context: ProcessingContext) -> ProcessingResult:
        """Process the content and return result."""
        ...

    @property
    def name(self) -> str:
        """Get processor name."""
        ...


class BasePreprocessor(ABC):
    """Abstract base class for preprocessors."""
    
    def __init__(self, config: dict[str, Any] | None = None):
        self._config = config or {}
        self._id = uuid4()
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the processor name."""
        pass
    
    @abstractmethod
    def _process_impl(self, context: ProcessingContext) -> ProcessingResult:
        """Implementation of processing logic."""
        pass
    
    def process(self, context: ProcessingContext) -> ProcessingResult:
        """Process content with error handling and logging."""
        try:
            return self._process_impl(context)
        except Exception as e:
            raise ProcessingError(
                f"Error in {self.name}: {str(e)}"
            ) from e


class Pipeline(BaseModel):
    """Represents a preprocessing pipeline."""
    
    name: str
    description: str | None = None
    processors: list[IPreprocessor]
    
    def process(self, content: str, content_type: str = "text/plain") -> ProcessingResult:
        """Run content through the pipeline."""
        context = ProcessingContext(
            content_id=uuid4(),
            content_type=content_type,
            original_content=content
        )
        
        current_result = ProcessingResult(content=content)
        
        for processor in self.processors:
            try:
                context.original_content = current_result.content
                current_result = processor.process(context)
            except ProcessingError as e:
                # Log error but continue pipeline
                current_result.metadata[f"error_{processor.name}"] = str(e)
        
        return current_result
