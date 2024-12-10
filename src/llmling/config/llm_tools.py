"""Wrapper for tools supplied to the LLM."""

from collections.abc import Callable, Sequence
from typing import Any

from llmling.config.runtime import RuntimeConfig
from llmling.tools.base import LLMCallableTool


class LLMTools:
    """Contains wrapped RuntimeConfig methods for better usage by LLMs.

    Adapts docstrings and return types.
    """

    def __init__(self, runtime: RuntimeConfig):
        self.runtime = runtime

    async def load_resource(self, name: str) -> dict[str, Any]:
        """Load the content of a resource by its name.

        Use this tool to load the actual content of a resource. First use
        get_resources() to see what's available, then provide the resource name
        from that list.

        Args:
            name: Name of the resource to load (must match a name from get_resources())

        Returns:
            The loaded resource including its content and metadata.

        Raises:
            ResourceError: If the resource doesn't exist or can't be loaded.
        """
        resource = await self.runtime.load_resource(name)
        return resource.model_dump()

    def get_resources(self) -> list[dict]:
        """List all available resources and their metadata.

        This tool returns information about all resources that can be loaded, including:
        - name: The identifier to use with load_resource
        - description: What the resource contains
        - mimeType: The type of content (e.g., text/markdown, application/json)

        Returns:
            List of resources with their complete metadata.
        """
        return [i.model_dump(exclude={"uri"}) for i in self.runtime.get_resources()]

    async def register_tool(
        self,
        name: str,
        function: str | Callable[..., Any],
        description: str | None = None,
    ) -> str:
        """Register a new tool from a function or import path.

        This tool can register a callable from an import path as string
        (e.g. "webbrowser.open")

        Args:
            name: Name for the new tool
            function: Function to register (callable or import path)
            description: Optional description override (uses function docstring if None)

        Returns:
            Message confirming the tool registration
        """
        return await self.runtime.register_tool(function, name, description)

    async def register_code_tool(
        self,
        name: str,
        code: str,
        description: str | None = None,
    ) -> str:
        """Register a new tool from Python code.

        The provided code should define a function that will be converted into a tool.
        The function's docstring will be used as the tool's description if no description
        is provided.

        Args:
            name: Name for the new tool
            code: Python code defining the tool function
            description: Optional description override (uses function docstring if None)

        Returns:
            Message confirming the tool registration
        """
        return await self.runtime.register_code_tool(name, code, description)

    def get_tools(self) -> Sequence[LLMCallableTool]:
        """Get all registered tools.

        Returns:
            List of all tools
        """
        return self.runtime.get_tools()

    async def install_package(
        self,
        package: str,
    ) -> str:
        """Install a Python package.

        This allows installing packages needed for tool functionality.
        Package specifications follow PIP format (e.g. "requests>=2.28.0").

        Args:
            package: Package specification to install (e.g. "requests", "pillow>=10.0.0")

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
            self.get_tools,
            self.install_package,
        ]
        return [LLMCallableTool.from_callable(fn) for fn in fns]  # type: ignore
