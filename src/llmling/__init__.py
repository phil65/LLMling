from __future__ import annotations

from llmling.context import (
    ContextLoader,
    LoadedContext,
    default_registry as context_registry,
)
from llmling.core.exceptions import (
    LLMLingError,
    ConfigError,
    ContextError,
    LoaderError,
    ProcessorError,
    LLMError,
    TaskError,
)
from llmling.llm.registry import default_registry as llm_registry
from llmling.processors.registry import ProcessorRegistry
from llmling.task.executor import TaskExecutor
from llmling.task.manager import TaskManager


__version__ = "0.1.1"

__all__ = [
    # Core components
    "ContextLoader",
    "LoadedContext",
    "ProcessorRegistry",
    "TaskExecutor",
    "TaskManager",
    # Default registries
    "context_registry",
    "llm_registry",
    # Exceptions
    "LLMLingError",
    "ConfigError",
    "ContextError",
    "LoaderError",
    "ProcessorError",
    "LLMError",
    "TaskError",
]

# llmling/
# ├── src/
# │   └── llmling/
# │       ├── __init__.py                 # Main package exports
# │       ├── cli.py                      # Command-line interface
# │       │
# │       ├── core/                       # Core components
# │       │   ├── __init__.py
# │       │   ├── exceptions.py           # Central exception hierarchy
# │       │   ├── log.py             # Logging configuration
# │       │   └── types.py               # Common type definitions
# │       │
# │       ├── config/                     # Configuration management
# │       │   ├── __init__.py
# │       │   ├── models.py              # Configuration models
# │       │   ├── manager.py             # Configuration management
# │       │   └── validation.py          # Configuration validation
# │       │
# │       ├── context/                    # Context handling
# │       │   ├── __init__.py
# │       │   ├── base.py                # Base context classes
# │       │   ├── registry.py            # Context loader registry
# │       │   └── loaders/               # Context loaders
# │       │       ├── __init__.py
# │       │       ├── path.py
# │       │       ├── text.py
# │       │       ├── cli.py
# │       │       ├── source.py
# │       │       └── callable.py
# │       │
# │       ├── processors/                 # Content processing
# │       │   ├── __init__.py
# │       │   ├── base.py                # Base processor classes
# │       │   ├── registry.py            # Processor registry
# │       │   └── implementations/       # Processor implementations
# │       │       ├── __init__.py
# │       │       ├── function.py
# │       │       └── template.py
# │       │
# │       ├── llm/                       # LLM integration
# │       │   ├── __init__.py
# │       │   ├── base.py                # Base LLM classes
# │       │   ├── registry.py            # LLM provider registry
# │       │   └── providers/             # LLM providers
# │       │       ├── __init__.py
# │       │       └── litellm.py
# │       │
# │       ├── task/                      # Task management
# │       │   ├── __init__.py
# │       │   ├── models.py              # Task models
# │       │   ├── executor.py            # Task execution
# │       │   ├── manager.py             # Task management
# │       │   └── concurrent.py          # Concurrent execution
# │       │
# │       └── utils/                     # Utilities
# │           ├── __init__.py
# │           ├── importing.py           # Import utilities
# │           └── calling.py             # Callable utilities
# │
# ├── tests/                             # Test suite
# │   ├── __init__.py
# │   ├── test_config.py
# │   ├── test_context.py
# │   ├── test_processors.py
# │   ├── test_llm.py
# │   ├── test_task.py
# │   └── test_concurrent.py
# │
# ├── examples/                          # Example usage
# ├── docs/                             # Documentation
# └── resources/                        # Resource files
