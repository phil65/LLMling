"""Tests for the processor system."""

from __future__ import annotations

import pytest

from llmling.core import exceptions
from llmling.core.typedefs import ProcessingStep
from llmling.processors.base import Processor, ProcessorConfig
from llmling.processors.registry import ProcessorRegistry
from llmling.resources.models import ProcessingContext


# Test data
SAMPLE_TEXT = "Hello, World!"
REVERSED_TEXT = SAMPLE_TEXT[::-1]
UPPER_TEXT = SAMPLE_TEXT.upper()
REVERSE_IMPORT = "llmling.testing.processors.reverse_text"
UPPERCASE_IMPORT = "llmling.testing.processors.uppercase_text"
APPEND_IMPORT = "llmling.testing.processors.append_text"
FAILING_IMPORT = "llmling.testing.processors.failing_processor"


# Test fixtures
@pytest.fixture
def registry() -> ProcessorRegistry:
    """Create a test processor registry."""
    return ProcessorRegistry()


@pytest.mark.asyncio
async def test_processor_pipeline(registry: ProcessorRegistry) -> None:
    """Test complete processor pipeline."""
    # Register processors
    registry.register("upper", ProcessorConfig(import_path=UPPERCASE_IMPORT))
    registry.register("append", ProcessorConfig(import_path=APPEND_IMPORT))

    # Define processing steps
    steps = [
        ProcessingStep(name="upper"),
        ProcessingStep(name="append", kwargs={"suffix": "!!!"}),
    ]

    # Process text
    try:
        await registry.startup()
        result = await registry.process("hello", steps)

        assert result.content == "HELLO!!!"
        assert result.original_content == "hello"
        assert "function" in result.metadata
    finally:
        await registry.shutdown()


@pytest.mark.asyncio
async def test_single_processor() -> None:
    """Test single processor execution."""
    processor = Processor(ProcessorConfig(import_path=REVERSE_IMPORT))

    try:
        await processor.startup()
        ctx = ProcessingContext(original_content=SAMPLE_TEXT, current_content=SAMPLE_TEXT)
        result = await processor.process(ctx)

        assert result.content == REVERSED_TEXT
        assert result.metadata["function"] == REVERSE_IMPORT
        assert not result.metadata["is_async"]
    finally:
        await processor.shutdown()


@pytest.mark.asyncio
async def test_processor_async() -> None:
    """Test asynchronous processor."""
    processor = Processor(
        ProcessorConfig(import_path="llmling.testing.processors.async_reverse_text")
    )

    try:
        await processor.startup()
        ctx = ProcessingContext(original_content=SAMPLE_TEXT, current_content=SAMPLE_TEXT)
        result = await processor.process(ctx)

        assert result.content == REVERSED_TEXT
        assert result.metadata["is_async"]
    finally:
        await processor.shutdown()


@pytest.mark.asyncio
async def test_processor_error() -> None:
    """Test processor error handling."""
    processor = Processor(ProcessorConfig(import_path=FAILING_IMPORT))

    await processor.startup()
    ctx = ProcessingContext(original_content=SAMPLE_TEXT, current_content=SAMPLE_TEXT)
    with pytest.raises(exceptions.ProcessorError, match="Processing failed"):
        await processor.process(ctx)


@pytest.mark.asyncio
async def test_registry_sequential_processing(registry: ProcessorRegistry) -> None:
    """Test sequential processing."""
    registry.register("reverse", ProcessorConfig(import_path=REVERSE_IMPORT))

    await registry.startup()
    try:
        steps = [ProcessingStep(name="reverse")]
        result = await registry.process("hello", steps)
        assert result.content == "olleh"
    finally:
        await registry.shutdown()


@pytest.mark.asyncio
async def test_registry_optional_step(registry: ProcessorRegistry) -> None:
    """Test optional step handling."""
    registry.register("fail", ProcessorConfig(import_path=FAILING_IMPORT))
    registry.register("reverse", ProcessorConfig(import_path=REVERSE_IMPORT))

    steps = [
        ProcessingStep(name="fail", required=False),
        ProcessingStep(name="reverse"),
    ]

    result = await registry.process(SAMPLE_TEXT, steps)
    assert result.content == REVERSED_TEXT


@pytest.mark.asyncio
async def test_registry_error_handling(registry: ProcessorRegistry) -> None:
    """Test registry error handling."""
    registry.register(
        "fail",
        ProcessorConfig(import_path=FAILING_IMPORT),
    )
    steps = [ProcessingStep(name="fail")]
    with pytest.raises(exceptions.ProcessorError):
        await registry.process(SAMPLE_TEXT, steps)


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
