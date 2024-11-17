"""Tests for the processor system."""

from __future__ import annotations

import asyncio

import pytest

from llmling.context.models import ProcessingContext
from llmling.core import exceptions
from llmling.core.typedefs import ProcessingStep
from llmling.processors.base import ProcessorConfig
from llmling.processors.implementations.function import FunctionProcessor
from llmling.processors.implementations.template import TemplateProcessor
from llmling.processors.registry import ProcessorRegistry


# Test data
SAMPLE_TEXT = "Hello, World!"
REVERSED_TEXT = "!dlroW ,olleH"
UPPER_TEXT = "HELLO, WORLD!"


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
    return ProcessorConfig(
        type="function",
        import_path="llmling.testing.processors.reverse_text",
    )


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
    registry.register(
        "upper",
        ProcessorConfig(
            type="function",
            import_path="llmling.testing.processors.uppercase_text",
        ),
    )
    registry.register(
        "append",
        ProcessorConfig(
            type="function",
            import_path="llmling.testing.processors.append_text",
        ),
    )

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
    config = ProcessorConfig(
        type="function",
        import_path="llmling.testing.processors.reverse_text",
    )

    processor = FunctionProcessor(config)

    try:
        await processor.startup()
        result = await processor.process(
            ProcessingContext(
                original_content=SAMPLE_TEXT,
                current_content=SAMPLE_TEXT,
                metadata={},
                kwargs={},
            ),
        )

        assert result.content == REVERSED_TEXT
        assert result.metadata["function"] == "llmling.testing.processors.reverse_text"
        assert not result.metadata["is_async"]
    finally:
        await processor.shutdown()


@pytest.fixture
def initialized_registry(registry: ProcessorRegistry) -> ProcessorRegistry:
    """Create and initialize a test processor registry."""
    # Register configurations
    registry.register(
        "reverse",
        ProcessorConfig(
            type="function",
            import_path="llmling.testing.processors.reverse_text",
        ),
    )
    registry.register(
        "reverse1",
        ProcessorConfig(
            type="function",
            import_path="llmling.testing.processors.reverse_text",
        ),
    )
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

    result = await processor.process(
        ProcessingContext(
            original_content=SAMPLE_TEXT,
            current_content=SAMPLE_TEXT,
            metadata={},
            kwargs={},
        ),
    )
    assert result.content == REVERSED_TEXT

    await processor.shutdown()


@pytest.mark.asyncio
async def test_processor_validation(function_config: ProcessorConfig) -> None:
    """Test processor result validation."""
    function_config.validate_output = True
    processor = FunctionProcessor(function_config)

    await processor.startup()
    result = await processor.process(
        ProcessingContext(
            original_content=SAMPLE_TEXT,
            current_content=SAMPLE_TEXT,
            metadata={},
            kwargs={},
        ),
    )
    assert result.content == REVERSED_TEXT


# Function processor tests
@pytest.mark.asyncio
async def test_function_processor_sync() -> None:
    """Test synchronous function processor."""
    config = ProcessorConfig(
        type="function",
        import_path="llmling.testing.processors.reverse_text",
    )
    processor = FunctionProcessor(config)

    await processor.startup()
    result = await processor.process(
        ProcessingContext(
            original_content=SAMPLE_TEXT,
            current_content=SAMPLE_TEXT,
            metadata={},
            kwargs={},
        ),
    )

    assert result.content == REVERSED_TEXT
    assert result.metadata["function"] == "llmling.testing.processors.reverse_text"
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
    result = await processor.process(
        ProcessingContext(
            original_content=SAMPLE_TEXT,
            current_content=SAMPLE_TEXT,
            metadata={},
            kwargs={},
        ),
    )

    assert result.content == REVERSED_TEXT
    assert result.metadata["function"] == "llmling.testing.processors.async_reverse_text"
    assert result.metadata["is_async"]


@pytest.mark.asyncio
async def test_function_processor_error() -> None:
    """Test function processor error handling."""
    config = ProcessorConfig(
        type="function",
        name="failing_processor",
        import_path="llmling.testing.processors.failing_processor",
    )
    processor = FunctionProcessor(config)

    await processor.startup()
    with pytest.raises(exceptions.ProcessorError, match="Function execution failed"):
        await processor.process(
            ProcessingContext(
                original_content=SAMPLE_TEXT,
                current_content=SAMPLE_TEXT,
                metadata={},
                kwargs={},
            ),
        )


# Template processor tests
@pytest.mark.asyncio
async def test_template_processor_basic(template_config: ProcessorConfig) -> None:
    """Test basic template processing."""
    processor = TemplateProcessor(template_config)

    await processor.startup()
    context = ProcessingContext(
        original_content=SAMPLE_TEXT,
        current_content=SAMPLE_TEXT,
        metadata={},
        kwargs={"extra": "value"},
    )

    result = await processor.process(context)

    assert result.content == f"Processed: {SAMPLE_TEXT}"
    assert "content" in result.metadata["template_vars"]
    assert "extra" in result.metadata["template_vars"]


@pytest.mark.asyncio
async def test_template_processor_error() -> None:
    """Test template processor error handling."""
    config = ProcessorConfig(
        type="template",
        template="{{ undefined_var + 123 }}",  # Force a template error
    )
    processor = TemplateProcessor(config)

    await processor.startup()
    with pytest.raises(exceptions.ProcessorError):
        await processor.process(
            ProcessingContext(
                original_content=SAMPLE_TEXT,
                current_content=SAMPLE_TEXT,
                metadata={},
                kwargs={},
            ),
        )


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
    assert registry._active
    await registry.shutdown()
    assert not registry._active


@pytest.mark.asyncio
async def test_registry_sequential_processing(
    initialized_registry: ProcessorRegistry,
) -> None:
    """Test sequential processing."""
    await initialized_registry.startup()
    try:
        result = await initialized_registry.process(
            "hello",
            [ProcessingStep(name="reverse")],
        )
        assert result.content == "olleh"
    finally:
        await initialized_registry.shutdown()


@pytest.mark.asyncio
async def test_registry_parallel_processing(registry: ProcessorRegistry) -> None:
    """Test parallel processing."""
    # Register processors first
    registry.register(
        "reverse1",
        ProcessorConfig(
            type="function",
            import_path="llmling.testing.processors.reverse_text",
        ),
    )
    registry.register(
        "reverse2",
        ProcessorConfig(
            type="function",
            import_path="llmling.testing.processors.reverse_text",
        ),
    )

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
    registry.register(
        "reverse",
        ProcessorConfig(
            type="function",
            import_path="llmling.testing.processors.async_reverse_text",
        ),
    )

    steps = [ProcessingStep(name="reverse")]
    results = [result async for result in registry.process_stream(SAMPLE_TEXT, steps)]
    assert len(results) == 1
    assert results[0].content == REVERSED_TEXT


@pytest.mark.asyncio
async def test_registry_optional_step(registry: ProcessorRegistry) -> None:
    registry.register(
        "fail",
        ProcessorConfig(
            type="function",
            import_path="llmling.testing.processors.failing_processor",
        ),
    )
    registry.register(
        "reverse",
        ProcessorConfig(
            type="function",
            import_path="llmling.testing.processors.reverse_text",
        ),
    )

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
        ProcessorConfig(
            type="function",
            name="failing_processor",
            import_path=f"{__name__}.failing_processor",
        ),
    )

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
        processor_registry.register(
            "reverse",
            ProcessorConfig(
                type="function",
                import_path=f"{__name__}.sync_reverse",
            ),
        )
        processor_registry.register(
            "template1",
            ProcessorConfig(
                type="template",
                template="First: {{ content }}",
            ),
        )
        processor_registry.register(
            "template2",
            ProcessorConfig(
                type="template",
                template="Second: {{ content }}",
            ),
        )

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
