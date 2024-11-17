"""Factory for creating preprocessing pipelines."""

from typing import Any

from .base import IPreprocessor, Pipeline
from .processors import (CodeBlockExtractor, HTMLCleaner, RedundancyRemover,
                       SentenceSegmenter, TextNormalizer)


class PipelineFactory:
    """Factory for creating preprocessing pipelines."""
    
    _processor_registry: dict[str, type[IPreprocessor]] = {
        "text_normalizer": TextNormalizer,
        "html_cleaner": HTMLCleaner,
        "sentence_segmenter": SentenceSegmenter,
        "redundancy_remover": RedundancyRemover,
        "code_block_extractor": CodeBlockExtractor
    }
    
    @classmethod
    def register_processor(cls, name: str, processor_class: type[IPreprocessor]) -> None:
        """Register a new processor type."""
        cls._processor_registry[name] = processor_class
    
    @classmethod
    def create_processor(cls, name: str, config: dict[str, Any] | None = None) -> IPreprocessor:
        """Create a processor instance."""
        if name not in cls._processor_registry:
            msg = f"Unknown processor type: {name}"
            raise ValueError(msg)
            
        processor_class = cls._processor_registry[name]
        return processor_class(config)
    
    @classmethod
    def create_pipeline(cls, config: dict[str, Any]) -> Pipeline:
        """Create a pipeline from configuration."""
        processors = []
        
        for proc_config in config.get("processors", []):
            processor_type = proc_config.pop("type")
            processor = cls.create_processor(processor_type, proc_config)
            processors.append(processor)
            
        return Pipeline(
            name=config.get("name", "default"),
            description=config.get("description"),
            processors=processors
        )


def create_default_pipeline() -> Pipeline:
    """Create a default preprocessing pipeline."""
    config = {
        "name": "default",
        "description": "Default preprocessing pipeline",
        "processors": [
            {"type": "text_normalizer"},
            {"type": "html_cleaner"},
            {"type": "sentence_segmenter"},
            {"type": "redundancy_remover"},
            {"type": "code_block_extractor"}
        ]
    }
    return PipelineFactory.create_pipeline(config)
