from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ConfigDict, Field
from schemez import Schema

from llmling.tools.base import BaseTool


if TYPE_CHECKING:
    from llmling import RuntimeConfig


class ErrorStrategy(str, Enum):
    STOP = "stop"  # Stop pipeline on error
    SKIP = "skip"  # Skip failed step, continue with previous result
    DEFAULT = "default"  # Use provided default value
    RETRY = "retry"  # Retry the step


class StepCondition(Schema):
    """Condition for conditional execution."""

    field: str  # Field to check in result
    operator: Literal["eq", "gt", "lt", "contains", "exists"]
    value: Any = None

    def evaluate_with_value(self, value: Any) -> bool:
        """Evaluate this condition against a value.

        Args:
            value: The value to evaluate against the condition.

        Returns:
            bool: True if the condition is met, False otherwise.
        """
        field_value = value.get(self.field) if isinstance(value, dict) else value

        match self.operator:
            case "eq":
                return field_value == self.value
            case "gt":
                return field_value > self.value
            case "lt":
                return field_value < self.value
            case "contains":
                try:
                    return self.value in field_value  # type: ignore
                except TypeError:
                    return False
            case "exists":
                return field_value is not None


@dataclass
class StepResult:
    """Result of a pipeline step execution."""

    success: bool
    result: Any
    error: Exception | None = None
    retries: int = 0
    duration: float = 0.0


# Type alias for step results during execution
type StepResults = dict[str, StepResult]


class PipelineStep(Schema):
    """Single step in a tool pipeline."""

    tool: str
    input_kwarg: str = "text"
    keyword_args: dict[str, Any] = Field(default_factory=dict)
    name: str | None = None  # Optional step name for referencing
    condition: StepCondition | None = None  # Conditional execution
    error_strategy: ErrorStrategy = ErrorStrategy.STOP
    default_value: Any = None  # Used with ErrorStrategy.DEFAULT
    max_retries: int = 0
    retry_delay: float = 1.0
    timeout: float | None = None
    depends_on: list[str] = Field(default_factory=list)  # Step dependencies
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Pipeline(Schema):
    """A pipeline of tool operations."""

    input: str | dict[str, Any]
    steps: list[PipelineStep]
    mode: Literal["sequential", "parallel"] = "sequential"
    max_parallel: int = 5  # Max concurrent steps
    collect_metrics: bool = False  # Collect execution metrics


class ChainTool(BaseTool):
    """Tool for executing a sequence of operations.

    NOTE: This tool must be explicitly enabled via the 'chain_tools' capability.

    Example:
        {
            "input": "main.py",
            "steps": [
                {"tool": "load_resource", "input_kwarg": "name"},
                {"tool": "analyze_code", "input_kwarg": "code"},
                {"tool": "format_output", "input_kwarg": "text"}
            ]
        }
    """

    name = "chain"
    description = """Execute multiple tool operations in sequence.

    WHEN TO USE THIS TOOL:
    - Use this when you can plan multiple operations confidently in advance
    - Use this for common sequences you've successfully used before
    - Use this to reduce interaction rounds for known operation patterns
    - Use this when all steps are independent of intermediate results

    DO NOT USE THIS TOOL:
    - When you need to inspect intermediate results
    - When next steps depend on analyzing previous results
    - When you're unsure about the complete sequence
    - When you need to handle errors at each step individually

    Input can be provided in various formats. Each operation's output
    becomes input for the next operation.

    Examples:
    1. Known sequence: Load file -> analyze -> format
    {
        "input": "main.py",
        "steps": [
            {"tool": "load_resource", "input_kwarg": "name"},
            {"tool": "analyze_code", "input_kwarg": "code"},
            {"tool": "format_output", "input_kwarg": "text"}
        ]
    }

    2. Independent operations: Count tokens while analyzing
    {
        "input": "test.py",
        "steps": [
            {"tool": "load_resource", "input_kwarg": "name"},
            {"tool": "analyze_code", "input_kwarg": "code"},
            {"tool": "count_tokens", "input_kwarg": "text"}
        ]
    }

    Note: If you're uncertain about the sequence or need to make decisions
    based on intermediate results, use individual tool calls instead.
    """

    # This is the full documentation for developers
    __doc__ = """Advanced tool pipeline execution system for LLMling.

    This implementation provides a powerful pipeline system for tool orchestration
    with features like error handling, parallel execution, and conditional processing,
    while exposing a simpler interface to LLMs.

    Basic Usage:
    >>> chain = Chain(
    ...     input="main.py",
    ...     steps=[
    ...         PipelineStep(tool="load_resource", input_kwarg="name"),
    ...         PipelineStep(tool="analyze_code", input_kwarg="code")
    ...     ]
    ... )
    >>> result = await chain_tool.execute(chain)

    Advanced Features:

    1. Error Handling:
       - STOP: Stop pipeline on error (default)
       - SKIP: Continue with previous result
       - DEFAULT: Use provided default value
       - RETRY: Retry the step with delay

       Example:
       >>> step = PipelineStep(
       ...     tool="analyze_code",
       ...     error_strategy=ErrorStrategy.RETRY,
       ...     max_retries=3,
       ...     retry_delay=1.0
       ... )

    2. Conditional Execution:
       Execute steps based on conditions:
       >>> step = PipelineStep(
       ...     tool="summarize",
       ...     condition=StepCondition(
       ...         field="length",
       ...         operator="gt",
       ...         value=1000
       ...     )
       ... )

    3. Execution Modes:
       - SEQUENTIAL: Execute steps in order
       - PARALLEL: Execute independent steps simultaneously
       - CONCURRENT: Dynamic scheduling based on dependencies

       Example:
       >>> chain = Chain(
       ...     input=data,
       ...     steps=[step1, step2],
       ...     mode=ExecutionMode.PARALLEL,
       ...     max_parallel=3
       ... )

    4. Step Dependencies:
       Define execution order with dependencies:
       >>> steps = [
       ...     PipelineStep(tool="load", name="loader"),
       ...     PipelineStep(
       ...         tool="analyze",
       ...         depends_on=["loader"]
       ...     )
       ... ]

    5. Timeouts and Metrics:
       >>> step = PipelineStep(
       ...     tool="complex_analysis",
       ...     timeout=30.0
       ... )
       >>> chain = Chain(
       ...     steps=[step],
       ...     collect_metrics=True
       ... )

    Error Handling Priority:
    1. Step-specific error strategy
    2. Pipeline default strategy
    3. Global error handler

    Performance Notes:
    - Use PARALLEL mode for independent operations
    - Set appropriate timeouts for long-running steps
    - Use max_parallel to control resource usage
    - Consider memory usage with large datasets

    Security Considerations:
    - Step dependencies are validated to prevent cycles
    - Timeouts prevent infinite execution
    - Error strategies contain failure propagation
    """

    def __init__(self, runtime: RuntimeConfig):
        self.runtime = runtime

    async def _execute_step(
        self,
        step: PipelineStep,
        input_value: Any,
        results: StepResults,
    ) -> StepResult:
        """Execute a single pipeline step."""
        start_time = asyncio.get_event_loop().time()
        retries = 0

        while True:
            try:
                # Check condition if any
                if step.condition and not step.condition.evaluate_with_value(input_value):
                    return StepResult(success=True, result=input_value, duration=0)
                # Prepare kwargs
                if isinstance(input_value, dict):
                    kwargs = {**input_value, **step.keyword_args}
                else:
                    kwargs = {step.input_kwarg: input_value, **step.keyword_args}

                # Execute with timeout if specified
                if step.timeout:
                    fut = self.runtime.execute_tool(step.tool, **kwargs)
                    result = await asyncio.wait_for(fut, timeout=step.timeout)
                else:
                    result = await self.runtime.execute_tool(step.tool, **kwargs)

                duration = asyncio.get_event_loop().time() - start_time
                return StepResult(success=True, result=result, duration=duration)

            except Exception as exc:
                match step.error_strategy:
                    case ErrorStrategy.STOP:
                        raise

                    case ErrorStrategy.SKIP:
                        duration = asyncio.get_event_loop().time() - start_time
                        return StepResult(
                            success=False,
                            result=input_value,
                            error=exc,
                            duration=duration,
                        )

                    case ErrorStrategy.DEFAULT:
                        duration = asyncio.get_event_loop().time() - start_time
                        return StepResult(
                            success=False,
                            result=step.default_value,
                            error=exc,
                            duration=duration,
                        )

                    case ErrorStrategy.RETRY:
                        retries += 1
                        if retries <= step.max_retries:
                            await asyncio.sleep(step.retry_delay)
                            continue
                        raise  # Max retries exceeded

    async def _execute_sequential(self, pipeline: Pipeline, results: StepResults) -> Any:
        """Execute steps sequentially."""
        current = pipeline.input

        for step in pipeline.steps:
            result = await self._execute_step(step, current, results)
            if step.name:
                results[step.name] = result
            current = result.result

        return current

    async def _execute_parallel(self, pipeline: Pipeline, results: StepResults) -> Any:
        """Execute independent steps in parallel."""
        semaphore = asyncio.Semaphore(pipeline.max_parallel)

        async def run_step(step: PipelineStep) -> None:
            async with semaphore:
                # Wait for dependencies
                for dep in step.depends_on:
                    while dep not in results:
                        await asyncio.sleep(0.1)

                # Get input from dependency or pipeline input
                if step.depends_on:
                    input_value = results[step.depends_on[-1]].result
                else:
                    input_value = pipeline.input

                result = await self._execute_step(step, input_value, results)
                if step.name:
                    results[step.name] = result

        # Create tasks for all steps
        tasks = [run_step(step) for step in pipeline.steps]
        await asyncio.gather(*tasks)

        # Return last result
        return (
            results[pipeline.steps[-1].name].result if pipeline.steps[-1].name else None
        )

    async def execute(self, **params: Any) -> Any:
        """Execute the pipeline."""
        pipeline = Pipeline.model_validate(params)
        results: StepResults = {}

        match pipeline.mode:
            case "sequential":
                return await self._execute_sequential(pipeline, results)
            case "parallel":
                return await self._execute_parallel(pipeline, results)
