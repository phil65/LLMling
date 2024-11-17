"""Tool calling functionality for API calls."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar
from pydantic import BaseModel, Field, create_model, ValidationError
from enum import Enum

class ToolParameterType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"

class ToolParameter(BaseModel):
    """Definition of a tool parameter."""
    type: ToolParameterType
    description: str
    enum: list[Any] | None = None
    optional: bool = False
    default: Any | None = None

class ToolDefinition(BaseModel):
    """Configuration for a callable tool."""
    name: str
    """Name of the tool"""
    
    description: str
    """Description of what the tool does"""
    
    parameters: dict[str, ToolParameter]
    """Parameters expected by the tool"""
    
    implementation: str | None = None
    """Python import path to tool implementation class"""

    def to_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function schema."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    name: {
                        "type": param.type,
                        **({"enum": param.enum} if param.enum else {}),
                        "description": param.description
                    }
                    for name, param in self.parameters.items()
                },
                "required": [
                    name for name, param in self.parameters.items()
                    if not param.optional
                ]
            }
        }

class ToolResponse(BaseModel):
    """Base class for tool responses."""
    tool_name: str
    """Name of the tool that was called"""
    
    success: bool = True
    """Whether the tool call succeeded"""
    
    result: Any
    """Result data from the tool execution"""
    
    error: str | None = None
    """Error message if the tool call failed"""

class Tool(ABC):
    """Base class for callable tools."""
    
    name: ClassVar[str]
    """Name of the tool"""
    
    description: ClassVar[str]
    """Description of what the tool does"""
    
    parameters: ClassVar[dict[str, Any]]
    """Parameters expected by the tool"""
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResponse:
        """Execute the tool with the given parameters."""
        raise NotImplementedError

    @classmethod
    def get_parameter_model(cls) -> type[BaseModel]:
        """Get a Pydantic model for parameter validation."""
        return create_model(
            f"{cls.__name__}Params",
            **{
                name: (
                    param.get("type", str),
                    param.get("default", ...),
                )
                for name, param in cls.parameters.items()
            }
        )
    
    def validate_and_execute(self, **kwargs) -> ToolResponse:
        """Validate parameters and execute the tool."""
        model = self.get_parameter_model()
        try:
            params = model(**kwargs)
            return self.execute(**params.model_dump())
        except ValidationError as e:
            return ToolResponse(
                tool_name=self.name,
                success=False,
                result=None,
                error=str(e)
            )

class ToolRegistry:
    """Registry of available tools."""
    
    def __init__(self):
        self._tools: dict[str, type[Tool] | ToolDefinition] = {}
    
    def register(self, tool: type[Tool] | ToolDefinition) -> None:
        """Register a tool class or definition."""
        name = tool.name if isinstance(tool, ToolDefinition) else tool.name
        self._tools[name] = tool
    
    def get_tool(self, name: str) -> type[Tool] | ToolDefinition | None:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_definitions(self) -> list[dict[str, Any]]:
        """Get function definitions for all registered tools."""
        defs = []
        for tool in self._tools.values():
            if isinstance(tool, ToolDefinition):
                defs.append(tool.to_schema())
            else:
                # Convert Tool class parameters to OpenAI schema format
                props = {}
                required = []
                for name, param in tool.parameters.items():
                    props[name] = {
                        "type": param.get("type", "string"),
                        "description": param.get("description", ""),
                    }
                    if param.get("enum"):
                        props[name]["enum"] = param["enum"]
                    if not param.get("optional", False):
                        required.append(name)
                
                defs.append({
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": required
                    }
                })
        return defs

# Global registry instance
registry = ToolRegistry()

# Example tool implementation
class CurrentWeatherTool(Tool):
    name = "get_current_weather"
    description = "Get the current weather for a location"
    parameters = {
        "location": {
            "type": "string",
            "description": "City and state or country"
        },
        "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "optional": True,
            "default": "celsius"
        }
    }
    
    def execute(self, location: str, unit: str = "celsius") -> ToolResponse:
        # Simulated weather response
        return ToolResponse(
            tool_name=self.name,
            result={
                "temperature": 22.5 if unit == "celsius" else 72.5,
                "unit": unit,
                "condition": "sunny"
            }
        )

# Register example tool
registry.register(CurrentWeatherTool)

if __name__ == "__main__":
    # Example tool registration from definition
    weather_tool = ToolDefinition(
        name="weather_api",
        description="Get weather information from API",
        parameters={
            "city": ToolParameter(
                type=ToolParameterType.STRING,
                description="City name",
            ),
            "days": ToolParameter(
                type=ToolParameterType.NUMBER,
                description="Forecast days",
                optional=True,
                default=1
            )
        },
        implementation="myapp.weather.WeatherAPI"
    )
    registry.register(weather_tool)
    
    # Test built-in weather tool
    tool = CurrentWeatherTool()
    response = tool.execute(location="London", unit="celsius")
    print("Weather tool response:", response.model_dump_json(indent=2))
    
    # Print all registered tool definitions
    print("\nRegistered tools:")
    for tool_def in registry.get_definitions():
        print(f"- {tool_def['name']}: {tool_def['description']}")
