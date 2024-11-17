"""Concurrent task execution utilities."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import logfire

from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.task.models import TaskResult


if TYPE_CHECKING:
    from collections.abc import Sequence

    from llmling.task.manager import TaskManager


logger = get_logger(__name__)


@logfire.instrument("Executing concurrent tasks: {', '.join(templates)}")
async def execute_concurrent(
    manager: TaskManager,
    templates: Sequence[str],
    system_prompt: str | None = None,
    max_concurrent: int = 3,
    **kwargs: Any,
) -> list[TaskResult]:
    """Execute multiple task templates concurrently.

    Args:
        manager: Task manager instance
        templates: Template names to execute
        system_prompt: Optional system prompt
        max_concurrent: Maximum concurrent tasks
        **kwargs: Additional parameters for LLM

    Returns:
        List of task results

    Raises:
        TaskError: If execution fails
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_with_semaphore(template: str) -> TaskResult:
        async with semaphore:
            return await manager.execute_template(
                template,
                system_prompt=system_prompt,
                **kwargs,
            )

    try:
        return await asyncio.gather(
            *(execute_with_semaphore(t) for t in templates),
        )
    except Exception as exc:
        msg = "Concurrent execution failed"
        raise exceptions.TaskError(msg) from exc
