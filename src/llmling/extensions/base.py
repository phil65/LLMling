"""Base class for extension loaders."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from epregistry import EntryPointRegistry

from llmling.core.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Callable


logger = get_logger(__name__)
T = TypeVar("T")


class BaseExtensionLoader[T]:
    """Base class for extension loaders.

    Entry points are expected to be registered under 'llmling' with their type
    as the entry point name, e.g.:
    [project.entry-points.llmling]
    tools = "my_module:get_mcp_tools"
    prompts = "my_module:get_mcp_prompts"
    """

    component_type: ClassVar[str]
    converter: Callable[[Any], T]

    def __init__(self) -> None:
        """Initialize loader."""
        self.registry = EntryPointRegistry[Any]("llmling")

    def load_items(self, module_names: list[str]) -> dict[str, T]:
        """Load items from specified modules."""
        items = {}
        for module in module_names:
            try:
                # Just look for the component type entry point
                if entry_point := self.registry.get(self.component_type):
                    get_items = entry_point.load()
                    for item in get_items():
                        try:
                            converted = self.converter(item)
                            name = getattr(converted, "name", str(item))
                            items[name] = converted
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "Failed to load item from %s: %s",
                                module,
                                exc,
                            )
            except Exception:
                logger.exception("Failed to load module %s", module)
        return items
