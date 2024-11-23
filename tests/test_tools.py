from __future__ import annotations

import pytest

from llmling.tools.base import LLMCallableTool
from llmling.tools.exceptions import ToolError
from llmling.tools.registry import ToolRegistry


EXAMPLE_IMPORT = "llmling.testing.tools.example_tool"
FAILING_IMPORT = "llmling.testing.tools.failing_tool"
ANALYZE_IMPORT = "llmling.testing.tools.analyze_ast"


# Test fixtures
@pytest.fixture
def registry() -> ToolRegistry:
    """Create a fresh tool registry."""
    return ToolRegistry()


async def test_tool_registration_and_execution():
    """Test basic tool registration and execution flow."""
    registry = ToolRegistry()

    # Register a simple tool
    registry["test_tool"] = EXAMPLE_IMPORT

    # Verify tool is registered
    assert "test_tool" in registry.list_items()

    # Execute tool and verify result
    result = await registry.execute("test_tool", text="hello", repeat=2)
    assert result == "hellohello"


async def test_tool_registry_errors():
    """Test error handling in tool registry."""
    registry = ToolRegistry()

    # Test executing non-existent tool
    with pytest.raises(ToolError):
        await registry.execute("non_existent_tool")

    # Test duplicate registration
    registry["test_tool"] = EXAMPLE_IMPORT
    with pytest.raises(ToolError):
        registry["test_tool"] = EXAMPLE_IMPORT


async def test_tool_execution_with_invalid_params():
    """Test tool execution with invalid parameters."""
    registry = ToolRegistry()
    registry["test_tool"] = EXAMPLE_IMPORT

    # Test with missing required parameter and parameter with wrong type
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        await registry.execute("test_tool")
    with pytest.raises(TypeError, match="unsupported operand type"):
        await registry.execute("test_tool", text=None)


async def test_failing_tool():
    """Test handling of a tool that raises an exception."""
    registry = ToolRegistry()
    registry["failing_tool"] = FAILING_IMPORT

    with pytest.raises(ValueError, match="Intentional failure"):
        await registry.execute("failing_tool", text="any input")


# Test DynamicTool
class TestDynamicTool:
    def test_init(self) -> None:
        """Test tool initialization."""
        tool = LLMCallableTool.from_callable(
            EXAMPLE_IMPORT, name_override="name", description_override="desc"
        )
        assert tool.name == "name"
        assert tool.description == "desc"
        assert tool.import_path == EXAMPLE_IMPORT

    def test_default_name(self) -> None:
        """Test default name from import path."""
        tool = LLMCallableTool.from_callable(EXAMPLE_IMPORT)
        assert tool.name == "example_tool"

    def test_default_description(self) -> None:
        """Test default description from docstring."""
        tool = LLMCallableTool.from_callable(EXAMPLE_IMPORT)
        assert "repeats text" in tool.description.lower()

    def test_schema_generation(self) -> None:
        """Test schema generation from function signature."""
        tool = LLMCallableTool.from_callable(EXAMPLE_IMPORT)
        schema = tool.get_schema()

        assert schema["function"]["name"] == "example_tool"
        assert "text" in schema["function"]["parameters"]["properties"]
        assert "repeat" in schema["function"]["parameters"]["properties"]
        assert schema["function"]["parameters"]["required"] == ["text"]

    @pytest.mark.asyncio
    async def test_execution(self) -> None:
        """Test tool execution."""
        tool = LLMCallableTool.from_callable(EXAMPLE_IMPORT)
        result = await tool.execute(text="test", repeat=2)
        assert result == "testtest"

    @pytest.mark.asyncio
    async def test_execution_failure(self) -> None:
        """Test tool execution failure."""
        tool = LLMCallableTool.from_callable(FAILING_IMPORT)
        with pytest.raises(Exception, match="Intentional"):
            await tool.execute(text="test")


# Test ToolRegistry
class TestToolRegistry:
    def test_register_path(self, registry: ToolRegistry) -> None:
        """Test registering a tool by import path."""
        registry["custom_tool"] = EXAMPLE_IMPORT
        assert "custom_tool" in registry.list_items()

    def test_register_duplicate(self, registry: ToolRegistry) -> None:
        """Test registering duplicate tool names."""
        registry["tool1"] = EXAMPLE_IMPORT
        with pytest.raises(ToolError):
            registry["tool1"] = EXAMPLE_IMPORT

    def test_get_nonexistent(self, registry: ToolRegistry) -> None:
        """Test getting non-existent tool."""
        with pytest.raises(ToolError):
            registry.get("nonexistent")

    def test_list_items(self, registry: ToolRegistry) -> None:
        """Test listing registered tools."""
        registry["tool1"] = EXAMPLE_IMPORT
        registry["tool2"] = ANALYZE_IMPORT
        tools = registry.list_items()
        assert len(tools) == 2  # noqa: PLR2004
        assert "tool1" in tools
        assert "tool2" in tools

    @pytest.mark.asyncio
    async def test_execute(self, registry: ToolRegistry) -> None:
        """Test executing a registered tool."""
        registry["example_tool"] = EXAMPLE_IMPORT
        result = await registry.execute("example_tool", text="test", repeat=3)
        assert result == "testtesttest"

    @pytest.mark.asyncio
    async def test_execute_with_validation(self, registry: ToolRegistry) -> None:
        """Test tool execution with invalid parameters."""
        registry["analyze_ast"] = ANALYZE_IMPORT
        # Valid Python code
        code = "class Test: pass\ndef func(): pass"
        result = await registry.execute("analyze_ast", code=code)
        assert result["classes"] == 1
        assert result["functions"] == 1

        # Invalid Python code
        with pytest.raises(Exception, match="invalid syntax"):
            await registry.execute("analyze_ast", code="invalid python")

    def test_schema_generation(self, registry: ToolRegistry) -> None:
        """Test schema generation for registered tools."""
        registry["analyze_ast"] = ANALYZE_IMPORT
        schema = registry.get_schema("analyze_ast")

        assert "code" in schema["function"]["parameters"]["properties"]
        assert schema["function"]["parameters"]["required"] == ["code"]
        assert "Analyze Python code AST" in schema["function"]["description"]


# Integration tests
@pytest.mark.asyncio
async def test_tool_integration() -> None:
    """Test full tool workflow."""
    # Setup
    registry = ToolRegistry()
    registry["analyze"] = ANALYZE_IMPORT

    # Get schema
    schema = registry.get_schema("analyze")
    assert schema["function"]["name"] == "analyze_ast"
    # Execute tool
    code = """
class TestClass:
    def method1(self):
        pass
    def method2(self):
        pass
    """
    result = await registry.execute("analyze", code=code)
    assert result["classes"] == code.count("class ")
    assert result["functions"] == code.count("def ")


if __name__ == "__main__":
    pytest.main(["-vv"])
