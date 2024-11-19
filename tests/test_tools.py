from __future__ import annotations

import pytest

from llmling.tools.base import DynamicTool, ToolRegistry
from llmling.tools.exceptions import ToolError


# Test fixtures
@pytest.fixture
def registry() -> ToolRegistry:
    """Create a fresh tool registry."""
    return ToolRegistry()


# Test DynamicTool
class TestDynamicTool:
    def test_init(self) -> None:
        """Test tool initialization."""
        tool = DynamicTool(
            import_path="llmling.testing.tools.example_tool",
            name="custom_name",
            description="Custom description",
        )
        assert tool.name == "custom_name"
        assert tool.description == "Custom description"
        assert tool.import_path == "llmling.testing.tools.example_tool"

    def test_default_name(self) -> None:
        """Test default name from import path."""
        tool = DynamicTool("llmling.testing.tools.example_tool")
        assert tool.name == "example_tool"

    def test_default_description(self) -> None:
        """Test default description from docstring."""
        tool = DynamicTool("llmling.testing.tools.example_tool")
        assert "repeats text" in tool.description.lower()

    def test_schema_generation(self) -> None:
        """Test schema generation from function signature."""
        tool = DynamicTool("llmling.testing.tools.example_tool")
        schema = tool.get_schema()

        assert schema.type == "function"
        assert schema.function["name"] == "example_tool"
        assert "text" in schema.function["parameters"]["properties"]
        assert "repeat" in schema.function["parameters"]["properties"]
        assert schema.function["parameters"]["required"] == ["text"]

    @pytest.mark.asyncio
    async def test_execution(self) -> None:
        """Test tool execution."""
        tool = DynamicTool("llmling.testing.tools.example_tool")
        result = await tool.execute(text="test", repeat=2)
        assert result == "testtest"

    @pytest.mark.asyncio
    async def test_execution_failure(self) -> None:
        """Test tool execution failure."""
        tool = DynamicTool("llmling.testing.tools.failing_tool")
        with pytest.raises(Exception, match="test"):
            await tool.execute(text="test")


# Test ToolRegistry
class TestToolRegistry:
    def test_register_path(self, registry: ToolRegistry) -> None:
        """Test registering a tool by import path."""
        registry.register_path(
            "llmling.testing.tools.example_tool",
            name="custom_tool",
        )
        assert "custom_tool" in registry.list_tools()

    def test_register_duplicate(self, registry: ToolRegistry) -> None:
        """Test registering duplicate tool names."""
        registry.register_path("llmling.testing.tools.example_tool", name="tool1")
        with pytest.raises(ToolError):
            registry.register_path("llmling.testing.tools.example_tool", name="tool1")

    def test_get_nonexistent(self, registry: ToolRegistry) -> None:
        """Test getting non-existent tool."""
        with pytest.raises(ToolError):
            registry["nonexistent"]

    def test_list_tools(self, registry: ToolRegistry) -> None:
        """Test listing registered tools."""
        registry.register_path("llmling.testing.tools.example_tool", name="tool1")
        registry.register_path("llmling.testing.tools.analyze_ast", name="tool2")
        tools = registry.list_tools()
        assert len(tools) == 2  # noqa: PLR2004
        assert "tool1" in tools
        assert "tool2" in tools

    @pytest.mark.asyncio
    async def test_execute(self, registry: ToolRegistry) -> None:
        """Test executing a registered tool."""
        registry.register_path("llmling.testing.tools.example_tool")
        result = await registry.execute("example_tool", text="test", repeat=3)
        assert result == "testtesttest"

    @pytest.mark.asyncio
    async def test_execute_with_validation(self, registry: ToolRegistry) -> None:
        """Test tool execution with invalid parameters."""
        registry.register_path("llmling.testing.tools.analyze_ast")

        # Valid Python code
        result = await registry.execute(
            "analyze_ast",
            code="class Test: pass\ndef func(): pass",
        )
        assert result["classes"] == 1
        assert result["functions"] == 1

        # Invalid Python code
        with pytest.raises(Exception, match="invalid syntax"):
            await registry.execute("analyze_ast", code="invalid python")

    def test_schema_generation(self, registry: ToolRegistry) -> None:
        """Test schema generation for registered tools."""
        registry.register_path(
            "llmling.testing.tools.analyze_ast",
            description="Custom description",
        )
        schema = registry.get_schema("analyze_ast")

        assert schema.type == "function"
        assert "code" in schema.function["parameters"]["properties"]
        assert schema.function["parameters"]["required"] == ["code"]
        assert schema.function["description"] == "Custom description"


# Integration tests
@pytest.mark.asyncio
async def test_tool_integration() -> None:
    """Test full tool workflow."""
    # Setup
    registry = ToolRegistry()
    registry.register_path(
        "llmling.testing.tools.analyze_ast",
        name="analyze",
        description="Analyze Python code",
    )

    # Get schema
    schema = registry.get_schema("analyze")
    assert schema.type == "function"

    # Execute tool
    code = """
class TestClass:
    def method1(self):
        pass
    def method2(self):
        pass
    """
    result = await registry.execute("analyze", code=code)

    assert result["classes"] == 1
    assert result["functions"] == 2  # noqa: PLR2004
