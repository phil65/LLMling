"""Wrapper for tools supplied to the LLM."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmling.tools.base import LLMCallableTool


if TYPE_CHECKING:
    from llmling.config.runtime import RuntimeConfig


class LLMTools:
    """Contains wrapped RuntimeConfig methods for better usage by LLMs.

    Adapts docstrings and return types.
    """

    def __init__(self, runtime: RuntimeConfig):
        self.runtime = runtime

    async def load_resource(self, name: str) -> dict[str, Any]:
        """Load the content of a resource to use in the current interaction.

        Use this to access content from available resources that you want
        to analyze or work with.

        Args:
            name: Name of an available resource to load

        Returns:
            The resource content and metadata
        """
        resource = await self.runtime.load_resource(name)
        return resource.model_dump()

    def get_resources(self) -> list[dict]:
        """List all resources that are available for loading.

        Use this to discover what resources you can access using load_resource().

        Returns:
            List of available resources with their descriptions
        """
        return [i.model_dump(exclude={"uri"}) for i in self.runtime.get_resources()]

    async def register_tool(
        self,
        name: str,
        function: str,
        description: str | None = None,
    ) -> str:
        """Register an importable function as a tool for future interactions.

        IMPORTANT: The registered tool will NOT be available in this current
        interaction. It can only be used in future requests.

        Args:
            function: Import path to the function (e.g. "json.dumps")
            name: Optional custom name for the tool (uses function name if not provided)
            description: What the tool does and how to use it

        Returns:
            Confirmation message about the registration
        """
        return await self.runtime.register_tool(function, name, description)

    async def register_code_tool(
        self,
        name: str,
        code: str,
        description: str | None = None,
    ) -> str:
        """Register new tool functionality for future interactions.

        IMPORTANT: The registered tool will NOT be available in this current
        interaction. It can only be used in future requests.

        Args:
            name: Identifying name for the new tool
            code: Python code that implements the tool
            description: What the tool does and how to use it

        Returns:
            Confirmation message about the registration
        """
        return await self.runtime.register_code_tool(name, code, description)

    async def install_package(
        self,
        package: str,
    ) -> str:
        """Install a Python package for future tool functionality.

        NOTE: Installed packages will only be available for tools in future requests,
        not in the current interaction.

        Args:
            package: Package specification to install (e.g. "requests>=2.28.0")

        Returns:
            Message confirming the installation
        """
        return await self.runtime.install_package(package)

    def get_llm_resource_tools(self) -> list[LLMCallableTool]:
        fns = [self.load_resource, self.get_resources]
        return [LLMCallableTool.from_callable(fn) for fn in fns]  # type: ignore

    def get_llm_tool_management_tools(self) -> list[LLMCallableTool]:
        fns = [
            self.register_tool,
            self.register_code_tool,
            self.install_package,
        ]
        return [LLMCallableTool.from_callable(fn) for fn in fns]  # type: ignore
