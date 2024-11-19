"""Tests for concurrent task execution."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest import mock

import pytest

from llmling.core import exceptions
from llmling.task.concurrent import execute_concurrent
from llmling.task.manager import TaskManager
from llmling.task.models import TaskResult


# Test data
TEMPLATE_NAMES = ["template1", "template2", "template3"]
SAMPLE_RESULTS = [
    TaskResult(
        content=f"Result {i}",
        model="test-model",
        context_metadata={},
        completion_metadata={},
    )
    for i in range(3)
]


# Fixtures
@pytest.fixture
def mock_manager() -> mock.MagicMock:
    """Create a mock task manager."""
    return mock.MagicMock(spec=TaskManager)


@pytest.fixture
def slow_manager() -> mock.MagicMock:
    """Create a mock task manager with delayed execution."""
    manager = mock.MagicMock(spec=TaskManager)

    async def slow_execute(
        template: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> TaskResult:
        await asyncio.sleep(0.1)
        return TaskResult(
            content=f"Result for {template}",
            model="test-model",
            context_metadata={},
            completion_metadata={"template": template},
        )

    manager.execute_template.side_effect = slow_execute
    return manager


# Tests
@pytest.mark.asyncio
async def test_concurrent_execution_basic(mock_manager: mock.MagicMock) -> None:
    """Test basic concurrent execution."""
    mock_manager.execute_template.side_effect = SAMPLE_RESULTS

    results = await execute_concurrent(mock_manager, TEMPLATE_NAMES, max_concurrent=2)

    assert len(results) == len(TEMPLATE_NAMES)
    assert all(isinstance(r, TaskResult) for r in results)
    assert mock_manager.execute_template.call_count == len(TEMPLATE_NAMES)


@pytest.mark.asyncio
async def test_concurrent_execution_concurrency_limit(
    slow_manager: mock.MagicMock,
) -> None:
    """Test that concurrency limit is respected."""
    max_concurrent = 2
    start_time = asyncio.get_event_loop().time()

    results = await execute_concurrent(
        slow_manager,
        TEMPLATE_NAMES,
        max_concurrent=max_concurrent,
    )

    elapsed = asyncio.get_event_loop().time() - start_time

    # With 3 tasks, max_concurrent=2, and 0.1s delay,
    # should take at least 0.2s (two batches)
    assert elapsed >= 0.2  # noqa: PLR2004
    assert len(results) == len(TEMPLATE_NAMES)


@pytest.mark.asyncio
async def test_concurrent_execution_with_system_prompt(
    mock_manager: mock.MagicMock,
) -> None:
    """Test concurrent execution with system prompt."""
    system_prompt = "Test prompt"
    mock_manager.execute_template.side_effect = SAMPLE_RESULTS

    await execute_concurrent(
        mock_manager,
        TEMPLATE_NAMES,
        system_prompt=system_prompt,
    )

    for call in mock_manager.execute_template.call_args_list:
        assert call.kwargs["system_prompt"] == system_prompt


@pytest.mark.asyncio
async def test_concurrent_execution_with_kwargs(
    mock_manager: mock.MagicMock,
) -> None:
    """Test concurrent execution with additional kwargs."""
    mock_manager.execute_template.side_effect = SAMPLE_RESULTS
    test_kwargs = {"param1": "value1", "param2": "value2"}

    await execute_concurrent(
        mock_manager,
        TEMPLATE_NAMES,
        **test_kwargs,
    )

    for call in mock_manager.execute_template.call_args_list:
        assert all(call.kwargs[k] == v for k, v in test_kwargs.items())


@pytest.mark.asyncio
async def test_concurrent_execution_empty_templates(
    mock_manager: mock.MagicMock,
) -> None:
    """Test concurrent execution with no templates."""
    results = await execute_concurrent(
        mock_manager,
        [],
        max_concurrent=2,
    )

    assert len(results) == 0
    mock_manager.execute_template.assert_not_called()


@pytest.mark.asyncio
async def test_concurrent_execution_error_handling(
    mock_manager: mock.MagicMock,
) -> None:
    """Test error handling in concurrent execution."""
    msg = "Test error"
    mock_manager.execute_template.side_effect = ValueError(msg)

    with pytest.raises(exceptions.TaskError) as exc_info:
        await execute_concurrent(
            mock_manager,
            TEMPLATE_NAMES,
            max_concurrent=2,
        )

    assert "Concurrent execution failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_concurrent_execution_partial_failure(
    mock_manager: mock.MagicMock,
) -> None:
    """Test handling of partial execution failures."""

    def side_effect(template: str, **kwargs: Any) -> TaskResult:
        if template == "template2":
            msg = "Test error"
            raise ValueError(msg)
        return SAMPLE_RESULTS[0]

    mock_manager.execute_template.side_effect = side_effect

    with pytest.raises(exceptions.TaskError):
        await execute_concurrent(mock_manager, TEMPLATE_NAMES, max_concurrent=2)


@pytest.mark.asyncio
async def test_concurrent_execution_cancellation(
    slow_manager: mock.MagicMock,
) -> None:
    """Test cancellation of concurrent execution."""
    # Create a task that we'll cancel
    task = asyncio.create_task(
        execute_concurrent(slow_manager, TEMPLATE_NAMES, max_concurrent=2),
    )

    # Let it start
    await asyncio.sleep(0.05)

    # Cancel it
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_concurrent_execution_order_preservation(
    mock_manager: mock.MagicMock,
) -> None:
    """Test that results preserve the order of input templates."""

    async def mock_execute(template: str, **kwargs: Any) -> TaskResult:
        return TaskResult(
            content=f"Result for {template}",
            model="test-model",
            context_metadata={},
            completion_metadata={"template": template},
        )

    mock_manager.execute_template.side_effect = mock_execute

    results = await execute_concurrent(
        mock_manager,
        TEMPLATE_NAMES,
        max_concurrent=2,
    )

    for template, result in zip(TEMPLATE_NAMES, results):
        assert result.completion_metadata["template"] == template


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
