from __future__ import annotations

from llmling.resources import (
    ResourceLoader,
    LoadedResource,
    default_registry as resource_registry,
)
from llmling.config.models import Config
from llmling.config.runtime import RuntimeConfig
from llmling.core.exceptions import (
    LLMLingError,
    ConfigError,
    ResourceError,
    LoaderError,
    ProcessorError,
    LLMError,
)
from llmling.processors.registry import ProcessorRegistry
from llmling.server import LLMLingServer


__version__ = "0.10.0"

__all__ = [
    "Config",
    "ConfigError",
    "LLMError",
    "LLMLingError",
    "LLMLingServer",
    "LoadedResource",
    "LoaderError",
    "ProcessorError",
    "ProcessorRegistry",
    "ResourceError",
    "ResourceLoader",
    "RuntimeConfig",
    "resource_registry",
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
# │       │   ├── manager.py             # Config management
# │       │   └── validation.py          # Config validation
# │       │
# │       ├── resources/                    # Resource handling
# │       │   ├── __init__.py
# │       │   ├── base.py                # Base resource classes
# │       │   ├── models.py              # Resource models
# │       │   ├── registry.py            # Resource registry
# │       │   └── loaders/               # Resource loaders
# │       │       ├── __init__.py
# │       │       ├── callable.py
# │       │       ├── cli.py
# │       │       ├── image.py
# │       │       ├── path.py
# │       │       ├── source.py
# │       │       └── text.py
# │       │
# │       ├── processors/                 # Resource processing
# │       │   ├── __init__.py
# │       │   ├── base.py                # Base processor classes
# │       │   ├── registry.py            # Processor registry
# │       │   └── implementations/       # Processor implementations
# │       │       ├── __init__.py
# │       │       ├── function.py
# │       │       └── template.py
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
# │       └── config_resources/                 # Configuration resources
# │           ├── test.yml              # Test configuration
# │           └── web_research.yml      # Web research config
# │
# ├── tests/                             # Test suite
# ├── examples/                          # Example usage
# ├── docs/                             # Documentation
# └── resources/                        # Resource files
