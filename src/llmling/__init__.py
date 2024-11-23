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


__version__ = "0.3.0"

__all__ = [
    "ConfigError",
    "ContextError",
    "ContextLoader",
    "LLMError",
    "LLMLingError",
    "LoadedContext",
    "LoaderError",
    "ProcessorError",
    "ProcessorRegistry",
    "TaskError",
    "TaskExecutor",
    "TaskManager",
    "context_registry",
    "llm_registry",
]

# llmling/
# ├── src/
# │   └── llmling/
# │       ├── __init__.py                 # Main package exports
# │       ├── client.py                   # High-level client interface
# │       │
# │       ├── core/                       # Core components
# │       │   ├── __init__.py
# │       │   ├── capabilities.py         # LLM model capabilities
# │       │   ├── descriptors.py          # Python descriptors
# │       │   ├── exceptions.py           # Exception hierarchy
# │       │   ├── log.py                  # Logging configuration
# │       │   ├── typedefs.py            # Common type definitions
# │       │   ├── utils.py               # Generic utilities
# │       │   └── baseregistry.py        # Base registry class
# │       │
# │       ├── config/                     # Configuration handling
# │       │   ├── __init__.py
# │       │   ├── models.py              # Configuration models
# │       │   ├── loading.py             # Config loading utilities
# │       │   ├── manager.py             # Config management
# │       │   └── validation.py          # Config validation
# │       │
# │       ├── context/                    # Context handling
# │       │   ├── __init__.py
# │       │   ├── base.py                # Base context classes
# │       │   ├── models.py              # Context models
# │       │   ├── registry.py            # Context registry
# │       │   └── loaders/               # Context loaders
# │       │       ├── __init__.py
# │       │       ├── callable.py
# │       │       ├── cli.py
# │       │       ├── image.py
# │       │       ├── path.py
# │       │       ├── source.py
# │       │       └── text.py
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
# │       ├── tools/                     # Tool system
# │       │   ├── __init__.py
# │       │   ├── base.py                # Base tool classes
# │       │   ├── actions.py             # Action definitions
# │       │   ├── browser.py             # Browser automation
# │       │   ├── code.py                # Code analysis
# │       │   └── exceptions.py          # Tool exceptions
# │       │
# │       ├── utils/                     # Utilities
# │       │   ├── __init__.py
# │       │   ├── importing.py           # Import utilities
# │       │   └── calling.py             # Callable utilities
# │       │
# │       ├── testing/                   # Testing utilities
# │       │   ├── __init__.py
# │       │   ├── processors.py          # Test processors
# │       │   └── tools.py               # Test tools
# │       │
# │       └── resources/                 # Configuration resources
# │           ├── system_tools.yml       # System tools config
# │           ├── test.yml              # Test configuration
# │           ├── vision_test.yml       # Vision testing config
# │           └── web_research.yml      # Web research config
# │
# ├── tests/                             # Test suite
# ├── examples/                          # Example usage
# ├── docs/                             # Documentation
# └── resources/                        # Resource files
