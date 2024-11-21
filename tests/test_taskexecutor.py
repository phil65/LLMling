from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from llmling.config.models import Context, TaskSettings, TextContext
from llmling.context.models import LoadedContext
from llmling.core import exceptions
from llmling.llm.base import (
    CompletionResult,
    LLMConfig,
    LLMProvider,
    Message,
)
from llmling.task.executor import TaskExecutor
from llmling.task.models import TaskContext, TaskProvider


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# Constants for test configuration
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 3
TEST_CONTENT = "test content"
TEST_MODEL = "test-model"
TEST_PROVIDER = "test-provider"
TEST_ROLE = "system"
TEST_MESSAGE = "test message"


class MockContextLoader:
    """Mock context loader for testing."""

    async def load(self, *args: Any, **kwargs: Any) -> LoadedContext:
        return LoadedContext(
            content=TEST_CONTENT, source_type="test", metadata={"test": "metadata"}
        )


class MockProcessorRegistry:
    """Mock processor registry."""

    async def process(self, *args: Any, **kwargs: Any) -> Any:
        return LoadedContext(content=TEST_CONTENT, source_type="test")


class MockLLMProvider(LLMProvider):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self.calls: list[tuple[list[Message], dict[str, Any]]] = []
        self._config = config  # Store for testing

    async def complete(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> CompletionResult:
        self.calls.append((messages, kwargs))
        return CompletionResult(
            content="test response",
            model=TEST_MODEL,
            metadata={"test": "metadata"},
        )

    async def complete_stream(
        self, messages: list[Message], **kwargs: Any
    ) -> AsyncIterator[CompletionResult]:
        yield CompletionResult(
            content="test chunk",
            model=TEST_MODEL,
            is_stream_chunk=True,
            metadata={"test": "metadata"},
        )


class FailingLLMProvider(LLMProvider):
    """Mock provider that fails."""

    async def complete(self, *args: Any, **kwargs: Any) -> CompletionResult:
        msg = "Test failure"
        raise exceptions.LLMError(msg)

    async def complete_stream(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[CompletionResult]:
        msg = "Test failure"
        raise exceptions.LLMError(msg)
        yield  # Required for type checking


class MockProviderRegistry:
    """Mock provider registry."""

    def create_provider(self, name: str, config: LLMConfig) -> LLMProvider:
        if name == "failing":
            return FailingLLMProvider(config)
        return MockLLMProvider(config)


@pytest.fixture
def context_registry() -> Any:
    """Mock context registry fixture."""

    class Registry:
        def get_loader(self, context: Context) -> Any:
            return MockContextLoader()

    return Registry()


@pytest.fixture
def processor_registry() -> MockProcessorRegistry:
    """Mock processor registry fixture."""
    return MockProcessorRegistry()


@pytest.fixture
def provider_registry() -> MockProviderRegistry:
    """Mock provider registry fixture."""
    return MockProviderRegistry()


@pytest.fixture
def executor(
    context_registry: Any,
    processor_registry: MockProcessorRegistry,
    provider_registry: MockProviderRegistry,
) -> TaskExecutor:
    """TaskExecutor fixture with mock dependencies."""
    return TaskExecutor(
        context_registry=context_registry,
        processor_registry=processor_registry,  # type: ignore
        provider_registry=provider_registry,  # type: ignore
        default_timeout=DEFAULT_TIMEOUT,
        default_max_retries=DEFAULT_MAX_RETRIES,
    )


@pytest.fixture
def task_context() -> TaskContext:
    """Sample task context fixture."""
    return TaskContext(
        context=TextContext(
            type="text", description="test context", content="test content"
        ),
        processors=[],
        inherit_tools=True,
    )


@pytest.fixture
def task_provider() -> TaskProvider:
    """Sample task provider fixture."""
    return TaskProvider(
        name=TEST_PROVIDER, model=TEST_MODEL, settings=TaskSettings(temperature=0.7)
    )


@pytest.mark.asyncio
async def test_execute_basic_task(
    executor: TaskExecutor, task_context: TaskContext, task_provider: TaskProvider
) -> None:
    """Test basic task execution."""
    result = await executor.execute(task_context, task_provider)

    assert result.content == "test response"
    assert result.model == TEST_MODEL
    assert result.context_metadata == {"test": "metadata"}
    assert result.completion_metadata == {"test": "metadata"}


@pytest.mark.asyncio
async def test_execute_with_system_prompt(
    executor: TaskExecutor, task_context: TaskContext, task_provider: TaskProvider
) -> None:
    """Test execution with system prompt."""
    result = await executor.execute(
        task_context, task_provider, system_prompt="Test system prompt"
    )
    assert result.content == "test response"


@pytest.mark.asyncio
async def test_execute_with_failing_provider(
    executor: TaskExecutor, task_context: TaskContext
) -> None:
    """Test handling of provider failures."""
    failing_provider = TaskProvider(name="failing", model=TEST_MODEL, settings=None)

    with pytest.raises(exceptions.TaskError):
        await executor.execute(task_context, failing_provider)


@pytest.mark.asyncio
async def test_execute_stream(
    executor: TaskExecutor, task_context: TaskContext, task_provider: TaskProvider
) -> None:
    """Test streaming execution."""
    chunks = [
        chunk async for chunk in executor.execute_stream(task_context, task_provider)
    ]

    assert len(chunks) == 1
    assert chunks[0].content == "test chunk"
    assert chunks[0].model == TEST_MODEL


@pytest.mark.asyncio
async def test_execute_stream_with_failing_provider(
    executor: TaskExecutor, task_context: TaskContext
) -> None:
    """Test streaming with failing provider."""
    failing_provider = TaskProvider(name="failing", model=TEST_MODEL, settings=None)
    with pytest.raises(exceptions.TaskError):
        _ = [_ async for _ in executor.execute_stream(task_context, failing_provider)]


@pytest.mark.asyncio
async def test_execute_with_custom_llm_config(
    executor: TaskExecutor, task_context: TaskContext
) -> None:
    """Test execution with custom LLM configuration."""
    # Create a new provider instance with custom settings
    settings = TaskSettings(temperature=0.5, max_tokens=100, top_p=0.9)
    provider = TaskProvider(name=TEST_PROVIDER, model=TEST_MODEL, settings=settings)
    result = await executor.execute(task_context, provider)
    assert result.content == "test response"


@pytest.mark.asyncio
async def test_concurrent_executions(
    executor: TaskExecutor, task_context: TaskContext, task_provider: TaskProvider
) -> None:
    """Test multiple concurrent executions."""
    i = 3
    tasks = [executor.execute(task_context, task_provider) for _ in range(i)]
    results = await asyncio.gather(*tasks)

    assert len(results) == i
    assert all(r.content == "test response" for r in results)


@pytest.mark.asyncio
async def test_execute_with_empty_messages(
    executor: TaskExecutor, task_context: TaskContext, task_provider: TaskProvider
) -> None:
    """Test handling of empty messages."""

    class EmptyContentLoader:
        async def load(self, *args: Any, **kwargs: Any) -> LoadedContext:
            return LoadedContext(content="", source_type="test")

    # Override context loader to return empty content
    executor.context_registry.get_loader = lambda x: EmptyContentLoader()  # type: ignore

    result = await executor.execute(task_context, task_provider)
    assert result.content == "test response"  # Should still work with empty content


def test_invalid_task_context(
    executor: TaskExecutor, task_provider: TaskProvider
) -> None:
    """Test handling of invalid context by simulating loader failure."""

    # Instead of trying to create an invalid context,
    # we'll test the error handling when the context loader fails
    class FailingContextLoader:
        async def load(self, *args: Any, **kwargs: Any) -> LoadedContext:
            msg = "Test failure"
            raise exceptions.LoaderError(msg)

    # Override context registry to return failing loader
    def get_failing_loader(*args: Any) -> Any:
        return FailingContextLoader()

    executor.context_registry.get_loader = get_failing_loader  # type: ignore

    text_ctx = TextContext(type="text", description="invalid ctx", content="some content")
    task_ctx = TaskContext(context=text_ctx, processors=[], inherit_tools=True)
    with pytest.raises(exceptions.TaskError):
        asyncio.run(executor.execute(task_ctx, task_provider))


# Additionally, let's add a test for validation errors
def test_validation_errors() -> None:
    """Test that validation errors are caught properly."""
    with pytest.raises(ValueError, match="text"):
        # Empty content should raise a validation error
        TextContext(type="text", description="test", content="")


# And a test for proper error chaining
@pytest.mark.asyncio
async def test_error_chaining(
    executor: TaskExecutor, task_provider: TaskProvider
) -> None:
    """Test that errors are properly chained through multiple levels."""

    class ChainTestLoader:
        async def load(self, *args: Any, **kwargs: Any) -> LoadedContext:
            try:
                msg = "Original error"
                raise ValueError(msg)  # noqa: TRY301
            except ValueError as e:
                msg = "Loader failed"
                raise exceptions.LoaderError(msg) from e

    executor.context_registry.get_loader = lambda x: ChainTestLoader()  # type: ignore
    text_ctx = TextContext(type="text", description="test", content="test content")
    valid_context = TaskContext(context=text_ctx, processors=[], inherit_tools=True)

    with pytest.raises(exceptions.TaskError) as exc_info:
        await executor.execute(valid_context, task_provider)

    # Get the full error chain
    error = exc_info.value
    assert str(error) == "Task execution failed"

    # First level cause
    cause1 = error.__cause__
    assert isinstance(cause1, exceptions.TaskError)
    assert str(cause1) == "Context loading failed"

    # Second level cause
    cause2 = cause1.__cause__
    assert isinstance(cause2, exceptions.LoaderError)
    assert str(cause2) == "Loader failed"

    # Third level cause (original error)
    cause3 = cause2.__cause__
    assert isinstance(cause3, ValueError)
    assert str(cause3) == "Original error"


# Add a test for simpler error chain
@pytest.mark.asyncio
async def test_simple_error_chain(
    executor: TaskExecutor, task_provider: TaskProvider
) -> None:
    """Test simpler error chain with direct loader error."""

    class SimpleErrorLoader:
        async def load(self, *args: Any, **kwargs: Any) -> LoadedContext:
            msg = "Direct loader error"
            raise exceptions.LoaderError(msg)

    executor.context_registry.get_loader = lambda x: SimpleErrorLoader()  # type: ignore
    text_ctx = TextContext(type="text", description="test", content="test content")
    valid_context = TaskContext(context=text_ctx, processors=[], inherit_tools=True)

    with pytest.raises(exceptions.TaskError) as exc_info:
        await executor.execute(valid_context, task_provider)

    # Check the error chain
    error = exc_info.value
    assert str(error) == "Task execution failed"

    cause = error.__cause__
    assert isinstance(cause, exceptions.TaskError)
    assert str(cause) == "Context loading failed"

    root_cause = cause.__cause__
    assert isinstance(root_cause, exceptions.LoaderError)
    assert str(root_cause) == "Direct loader error"


@pytest.mark.asyncio
async def test_custom_provider_settings(
    executor: TaskExecutor,
    task_context: TaskContext,
) -> None:
    """Test provider creation with custom settings."""
    # Create a provider with custom settings
    custom_settings = TaskSettings(temperature=0.8, max_tokens=100, top_p=0.95)
    provider = TaskProvider(
        name=TEST_PROVIDER,
        model=TEST_MODEL,
        settings=custom_settings,
    )

    result = await executor.execute(task_context, provider)

    assert result.content == "test response"
    assert result.model == TEST_MODEL


def test_prepare_messages(executor: TaskExecutor) -> None:
    """Test message preparation logic."""
    # Test with string content
    content = "Test user content"
    system_prompt = "You are a test assistant"
    messages = executor._prepare_messages(content, system_prompt)

    assert len(messages) == 2  # noqa: PLR2004
    assert messages[0].role == "system"
    assert messages[0].content == system_prompt
    assert messages[1].role == "user"
    assert messages[1].content == content

    # Test with LoadedContext
    loaded_context = LoadedContext(
        content="Test loaded content",
        source_type="test",
        metadata={},
    )
    messages = executor._prepare_messages(loaded_context, system_prompt)

    assert len(messages) == 2  # noqa: PLR2004
    assert messages[0].role == "system"
    assert messages[0].content == system_prompt
    assert messages[1].role == "user"
    assert messages[1].content == "Test loaded content"

    # Test without system prompt
    messages = executor._prepare_messages(content, None)
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == content


@pytest.mark.asyncio
async def test_provider_call_validation(
    executor: TaskExecutor,
    task_context: TaskContext,
) -> None:
    """Test that provider is called with correct parameters."""
    system_prompt = "Test system prompt"
    task_settings = TaskSettings(temperature=0.7, max_tokens=50)
    provider = TaskProvider(name=TEST_PROVIDER, model=TEST_MODEL, settings=task_settings)

    # Execute the task
    result = await executor.execute(
        task_context,
        provider,
        system_prompt=system_prompt,
    )

    assert result.content == "test response"
    assert result.model == TEST_MODEL


@pytest.mark.asyncio
async def test_provider_settings_validation(
    executor: TaskExecutor,
    task_context: TaskContext,
) -> None:
    """Test provider settings validation."""
    base_settings = TaskSettings(temperature=0.7, max_tokens=100, top_p=0.9)

    # Create provider with base settings
    provider = TaskProvider(name=TEST_PROVIDER, model=TEST_MODEL, settings=base_settings)

    # Execute with settings
    result = await executor.execute(task_context, provider)

    assert result.content == "test response"
    assert result.model == TEST_MODEL


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
