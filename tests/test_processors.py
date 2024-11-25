"""Tests for the processor system."""

from __future__ import annotations

import asyncio

import pytest

from llmling.core import exceptions
from llmling.core.typedefs import ProcessingStep
from llmling.processors.base import ProcessorConfig
from llmling.processors.implementations.function import FunctionProcessor
from llmling.processors.implementations.template import TemplateProcessor
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


# Test helpers
def sync_reverse(text: str) -> str:
    """Test helper to reverse text."""
    return text[::-1]


async def async_reverse(text: str) -> str:
    """Test helper to reverse text asynchronously."""
    await asyncio.sleep(0.1)
    return text[::-1]


async def failing_processor(text: str) -> str:
    """Test helper that fails."""
    msg = "Test failure"
    raise ValueError(msg)


# Test fixtures
@pytest.fixture
def function_config() -> ProcessorConfig:
    """Create a test function processor config."""
    return ProcessorConfig(type="function", import_path=REVERSE_IMPORT)


@pytest.fixture
def template_config() -> ProcessorConfig:
    """Create a test template processor config."""
    return ProcessorConfig(
        type="template",
        template="Processed: {{ content }}",
    )


@pytest.fixture
def registry() -> ProcessorRegistry:
    """Create and initialize a test processor registry."""
    return ProcessorRegistry()


@pytest.mark.asyncio
async def test_processor_pipeline(registry: ProcessorRegistry) -> None:
    """Test complete processor pipeline."""
    # Register processors
    cfg = ProcessorConfig(type="function", import_path=UPPERCASE_IMPORT)
    cfg_2 = ProcessorConfig(type="function", import_path=APPEND_IMPORT)
    registry.register("upper", cfg)
    registry.register("append", cfg_2)

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
async def test_function_processor() -> None:
    """Test function processor execution."""
    config = ProcessorConfig(type="function", import_path=REVERSE_IMPORT)

    processor = FunctionProcessor(config)

    try:
        await processor.startup()
        ctx = ProcessingContext(original_content=SAMPLE_TEXT, current_content=SAMPLE_TEXT)
        result = await processor.process(ctx)

        assert result.content == REVERSED_TEXT
        assert result.metadata["function"] == REVERSE_IMPORT
        assert not result.metadata["is_async"]
    finally:
        await processor.shutdown()


@pytest.fixture
def initialized_registry(registry: ProcessorRegistry) -> ProcessorRegistry:
    """Create and initialize a test processor registry."""
    # Register configurations
    cfg = ProcessorConfig(type="function", import_path=REVERSE_IMPORT)
    cfg_2 = ProcessorConfig(type="function", import_path=REVERSE_IMPORT)
    registry.register("reverse", cfg)
    registry.register("reverse1", cfg_2)
    return registry


@pytest.fixture
def processor_registry(registry: ProcessorRegistry) -> ProcessorRegistry:
    """Create a processor registry for complex tests."""
    return registry


# Base processor tests
@pytest.mark.asyncio
async def test_processor_lifecycle(function_config: ProcessorConfig) -> None:
    """Test processor startup and shutdown."""
    processor = FunctionProcessor(function_config)

    await processor.startup()
    assert processor.func is not None

    ctx = ProcessingContext(original_content=SAMPLE_TEXT, current_content=SAMPLE_TEXT)
    result = await processor.process(ctx)
    assert result.content == REVERSED_TEXT

    await processor.shutdown()


@pytest.mark.asyncio
async def test_processor_validation(function_config: ProcessorConfig) -> None:
    """Test processor result validation."""
    function_config.validate_output = True
    processor = FunctionProcessor(function_config)

    await processor.startup()
    ctx = ProcessingContext(original_content=SAMPLE_TEXT, current_content=SAMPLE_TEXT)
    result = await processor.process(ctx)
    assert result.content == REVERSED_TEXT


# Function processor tests
@pytest.mark.asyncio
async def test_function_processor_sync() -> None:
    """Test synchronous function processor."""
    config = ProcessorConfig(type="function", import_path=REVERSE_IMPORT)
    processor = FunctionProcessor(config)

    await processor.startup()
    ctx = ProcessingContext(original_content=SAMPLE_TEXT, current_content=SAMPLE_TEXT)
    result = await processor.process(ctx)

    assert result.content == REVERSED_TEXT
    assert result.metadata["function"] == REVERSE_IMPORT
    assert not result.metadata["is_async"]


@pytest.mark.asyncio
async def test_function_processor_async() -> None:
    """Test asynchronous function processor."""
    config = ProcessorConfig(
        type="function",
        name="async_reverse",
        import_path="llmling.testing.processors.async_reverse_text",
    )
    processor = FunctionProcessor(config)

    await processor.startup()
    ctx = ProcessingContext(original_content=SAMPLE_TEXT, current_content=SAMPLE_TEXT)
    result = await processor.process(ctx)

    assert result.content == REVERSED_TEXT
    assert result.metadata["function"] == "llmling.testing.processors.async_reverse_text"
    assert result.metadata["is_async"]


@pytest.mark.asyncio
async def test_function_processor_error() -> None:
    """Test function processor error handling."""
    config = ProcessorConfig(
        type="function",
        name="failing_processor",
        import_path=FAILING_IMPORT,
    )
    processor = FunctionProcessor(config)

    await processor.startup()
    ctx = ProcessingContext(original_content=SAMPLE_TEXT, current_content=SAMPLE_TEXT)
    with pytest.raises(exceptions.ProcessorError, match="Function execution failed"):
        await processor.process(ctx)


# Template processor tests
@pytest.mark.asyncio
async def test_template_processor_basic(template_config: ProcessorConfig) -> None:
    """Test basic template processing."""
    processor = TemplateProcessor(template_config)

    await processor.startup()
    context = ProcessingContext(
        original_content=SAMPLE_TEXT,
        current_content=SAMPLE_TEXT,
        kwargs={"extra": "value"},
    )

    result = await processor.process(context)

    assert result.content == f"Processed: {SAMPLE_TEXT}"
    assert "content" in result.metadata["template_vars"]
    assert "extra" in result.metadata["template_vars"]


@pytest.mark.asyncio
async def test_template_processor_error() -> None:
    """Test template processor error handling."""
    # Force a template error
    config = ProcessorConfig(type="template", template="{{ undefined_var + 123 }}")
    processor = TemplateProcessor(config)

    await processor.startup()
    ctx = ProcessingContext(original_content=SAMPLE_TEXT, current_content=SAMPLE_TEXT)
    with pytest.raises(exceptions.ProcessorError):
        await processor.process(ctx)


# Registry tests
@pytest.mark.asyncio
async def test_registry_lifecycle(registry: ProcessorRegistry) -> None:
    """Test registry startup and shutdown."""
    # Create config first, with all required fields
    config = ProcessorConfig(
        type="function",
        import_path=f"{__name__}.sync_reverse",
        # Don't set name directly, let the validator handle it
    )

    registry.register("reverse", config)
    await registry.startup()
    assert registry._initialized
    await registry.shutdown()
    assert not registry._initialized


@pytest.mark.asyncio
async def test_registry_sequential_processing(
    initialized_registry: ProcessorRegistry,
) -> None:
    """Test sequential processing."""
    await initialized_registry.startup()
    try:
        steps = [ProcessingStep(name="reverse")]
        result = await initialized_registry.process("hello", steps)
        assert result.content == "olleh"
    finally:
        await initialized_registry.shutdown()


@pytest.mark.asyncio
async def test_registry_parallel_processing(registry: ProcessorRegistry) -> None:
    """Test parallel processing."""
    # Register processors first
    cfg = ProcessorConfig(type="function", import_path=REVERSE_IMPORT)
    cfg_2 = ProcessorConfig(type="function", import_path=REVERSE_IMPORT)
    registry.register("reverse1", cfg)
    registry.register("reverse2", cfg_2)

    # Then start the registry
    await registry.startup()

    steps = [
        ProcessingStep(name="reverse1", parallel=True),
        ProcessingStep(name="reverse2", parallel=True),
    ]

    try:
        result = await registry.process(SAMPLE_TEXT, steps)
        assert REVERSED_TEXT in result.content
    finally:
        await registry.shutdown()


@pytest.mark.asyncio
async def test_registry_streaming(registry: ProcessorRegistry) -> None:
    p = "llmling.testing.processors.async_reverse_text"
    registry.register("reverse", ProcessorConfig(type="function", import_path=p))

    steps = [ProcessingStep(name="reverse")]
    results = [result async for result in registry.process_stream(SAMPLE_TEXT, steps)]
    assert len(results) == 1
    assert results[0].content == REVERSED_TEXT


@pytest.mark.asyncio
async def test_registry_optional_step(registry: ProcessorRegistry) -> None:
    cfg = ProcessorConfig(type="function", import_path=FAILING_IMPORT)
    registry.register("fail", cfg)
    cfg = ProcessorConfig(type="function", import_path=REVERSE_IMPORT)
    registry.register("reverse", cfg)

    steps = [
        ProcessingStep(name="fail", required=False),
        ProcessingStep(name="reverse"),
    ]

    result = await registry.process(SAMPLE_TEXT, steps)
    assert result.content == REVERSED_TEXT


@pytest.mark.asyncio
async def test_registry_error_handling(registry: ProcessorRegistry) -> None:
    """Test registry error handling."""
    p = f"{__name__}.failing_processor"
    cfg = ProcessorConfig(type="function", name="failing_processor", import_path=p)
    registry.register("fail", cfg)
    steps = [ProcessingStep(name="fail")]
    with pytest.raises(exceptions.ProcessorError):
        await registry.process(SAMPLE_TEXT, steps)


# Integration tests
@pytest.mark.asyncio
async def test_complex_processing_pipeline(processor_registry: ProcessorRegistry) -> None:
    """Test complex processing pipeline with sequential and parallel steps."""
    await processor_registry.startup()
    try:
        # Register processors
        cfg = ProcessorConfig(type="function", import_path=f"{__name__}.sync_reverse")
        processor_registry.register("reverse", cfg)
        cfg = ProcessorConfig(type="template", template="First: {{ content }}")
        processor_registry.register("template1", cfg)
        cfg = ProcessorConfig(type="template", template="Second: {{ content }}")
        processor_registry.register("template2", cfg)

        steps = [
            ProcessingStep(name="template1"),
            ProcessingStep(name="template2"),
            ProcessingStep(name="reverse"),
        ]

        result = await processor_registry.process("Hello, World!", steps)
        content = result.content

        # The content is reversed, so we need to reverse it back to check
        unreversed = content[::-1]
        assert any(marker in unreversed for marker in ["First:", "Second:"]), (
            f"Content (unreversed) does not contain expected markers: {unreversed}"
        )

        # Or we could check for the reversed markers
        assert any(marker in content for marker in [":tsriF", ":dnoceS"]), (
            f"Content does not contain reversed markers: {content}"
        )

    finally:
        await processor_registry.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
