"""Base class for tool collections."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from llmling.core.log import get_logger
from llmling.tools.base import LLMCallableTool


if TYPE_CHECKING:
    from collections.abc import Callable


logger = get_logger(__name__)


class ToolSet(ABC):
    """Container class for related tools."""

    @abstractmethod
    def get_tools(self) -> list[Callable[..., Any]]:
        """Get all tools from this toolset.

        Returns:
            List of callables to be converted to LLMCallableTools
        """
        raise NotImplementedError

    def get_llm_callable_tools(self) -> list[LLMCallableTool]:
        """Get all tools as LLMCallableTools.

        Returns:
            List of LLMCallableTools ready for registration

        Raises:
            ValueError: If a tool cannot be converted
        """
        tools = []
        for method in self.get_tools():
            try:
                tool = LLMCallableTool.from_callable(method)
                tools.append(tool)
            except Exception as exc:
                logger.exception("Failed to convert %s to LLMCallableTool", method)
                msg = f"Failed to convert {method.__name__} to LLMCallableTool"
                raise ValueError(msg) from exc
        return tools
