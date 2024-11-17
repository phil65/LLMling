"""Task execution and management system for LLM interactions.

This package provides a framework for defining, managing, and executing tasks using
Large Language Models (LLMs). It handles context loading, processing, and interaction
with LLM providers in a configurable and extensible way.

Key Components:
- TaskManager: Central component for managing and executing task templates
- TaskExecutor: Handles the actual execution of tasks with LLMs
- TaskContext: Configuration for task context and processing
- TaskProvider: Configuration for LLM provider settings
- TaskResult: Result of task execution

Usage Examples:

Basic task execution:
```python
result = await task_manager.execute_template(
    "my_template",
    system_prompt="You are a helpful assistant",
)
print(result.content)
```

Streaming execution:
```python
async for result in task_manager.execute_template_stream("my_template"):
    print(result.content)
```

Concurrent execution:
```python
results = await execute_concurrent(
    task_manager,
    templates=["template1", "template2"],
    max_concurrent=3,
)
```

Template Configuration:
```yaml
task_templates:
  my_template:
    context: my_context
    provider: gpt4
    inherit_tools: true
    settings:
      temperature: 0.7
      max_tokens: 1000
```
"""

from __future__ import annotations

from llmling.task.concurrent import execute_concurrent
from llmling.task.executor import TaskExecutor
from llmling.task.manager import TaskManager
from llmling.task.models import TaskContext, TaskProvider, TaskResult


__all__ = [
    # Core components
    "TaskManager",
    "TaskExecutor",
    # Models
    "TaskContext",
    "TaskProvider",
    "TaskResult",
    # Utilities
    "execute_concurrent",
]
