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
        """Load and access content from available resources.

        WHEN TO USE THIS TOOL:
        - When you need to read or analyze existing content
        - When you need to access documentation or code files
        - When you need reference material for a task

        DO NOT USE THIS TOOL:
        - When you already have the content in your context
        - When you're unsure if a resource exists
        - For writing or modifying content

        Args:
            name: Name of an available resource to load

        Examples:
            1. Loading a Python file:
               result = await load_resource("main.py")

            2. Loading documentation:
               docs = await load_resource("python_style_guide")
        """
        resource = await self.runtime.load_resource(name)
        return resource.model_dump()

    def get_resources(self) -> list[dict]:
        """Discover what resources are available for loading.

        WHEN TO USE THIS TOOL:
        - At the start of a task to understand available resources
        - When you need to find specific types of content
        - When you're unsure what resources exist
        - Before using load_resource to verify resource exists

        DO NOT USE THIS TOOL:
        - When you already know the resource exists
        - Repeatedly for the same information
        - Just to check if a single resource exists

        Returns:
            List of available resources with their descriptions

        Example:
            resources = await get_resources()
            # Returns: [
            #     {"name": "main.py", "description": "Main application code"},
            #     {"name": "docs/guide.md", "description": "User guide"}
            # ]
        """
        return [i.model_dump(exclude={"uri"}) for i in self.runtime.get_resources()]

    async def register_tool(
        self,
        name: str,
        function: str,
        description: str | None = None,
    ) -> str:
        """Register an importable function as a tool for future interactions.

        WHEN TO USE THIS TOOL:
        - When you identify a need for repeated functionality
        - When you find a Python function that would be useful as a tool
        - To make commonly used operations available as tools

        DO NOT USE THIS TOOL:
        - For one-off operations
        - When unsure about function safety
        - For functions that require complex setup
        - When similar tool already exists

        NOTE: Registered tools become available only in future interactions,
        not in the current one.

        Args:
            function: Import path to the function (e.g. "json.dumps")
            name: Optional custom name for the tool
            description: Optional description of what the tool does

        Example:
            await register_tool(
                "textwrap.dedent",
                name="format_code",
                description="Remove common leading whitespace from code"
            )
        """
        return await self.runtime.register_tool(function, name, description)

    async def register_code_tool(
        self,
        name: str,
        code: str,
        description: str | None = None,
    ) -> str:
        """Register new tool functionality from Python code.

        WHEN TO USE THIS TOOL:
        - When you need custom functionality not available in existing tools
        - When you need to optimize a specific operation
        - When you need to combine multiple operations into one tool

        DO NOT USE THIS TOOL:
        - For simple operations that existing tools can handle
        - When the operation is only needed once
        - When you're unsure about the code's safety or correctness
        - For code that requires external dependencies

        NOTE: The tool will only be available in future interactions.
        The code must be self-contained and define a single main function.

        Args:
            name: Name for the new tool
            code: Python code implementing the tool
            description: Optional description of what the tool does

        Example:
            await register_code_tool(
                name="word_count",
                code='''
                def word_count(text: str) -> dict[str, int]:
                    words = text.split()
                    return {"total": len(words)}
                ''',
                description="Count total words in text"
            )
        """
        return await self.runtime.register_code_tool(name, code, description)

    async def install_package(
        self,
        package: str,
    ) -> str:
        """Install a Python package for future tool functionality.

        WHEN TO USE THIS TOOL:
        - When registering a tool that requires a specific package
        - When you need functionality from a standard Python package
        - Before registering tools that have dependencies

        DO NOT USE THIS TOOL:
        - For untrusted or unknown packages
        - When the package is already installed
        - For packages with complex system requirements
        - When you're unsure if the package is needed

        NOTE: Installed packages only become available for tools in future
        interactions, not in the current one.

        Args:
            package: Package specification (e.g. "requests>=2.28.0")

        Example:
            await install_package("beautifulsoup4>=4.12.0")
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
