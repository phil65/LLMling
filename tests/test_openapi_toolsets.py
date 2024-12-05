from __future__ import annotations

import json

from openapi_spec_validator import validate
from openapi_spec_validator.exceptions import OpenAPISpecValidatorError
import pytest

from llmling.tools.openapi import OpenAPITools


PETSTORE_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Pet Store API", "version": "1.0.0"},
    "paths": {
        "/pet/{petId}": {
            "get": {
                "operationId": "get_pet",
                "summary": "Get pet by ID",
                "description": "Get pet by ID",
                "parameters": [
                    {
                        "name": "petId",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "integer"},
                        "description": "ID of pet to find",
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Pet found",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["id", "name"],
                                    "properties": {
                                        "id": {"type": "integer"},
                                        "name": {"type": "string"},
                                    },
                                }
                            }
                        },
                    }
                },
            }
        }
    },
    "servers": [
        {"url": "https://api.example.com", "description": "Pet store API server"}
    ],
}


@pytest.fixture
def mock_openapi_spec(tmp_path):
    """Set up OpenAPI spec mocking and local file."""
    try:
        validate(PETSTORE_SPEC)
        print("\nOpenAPI spec validation passed")
    except OpenAPISpecValidatorError as e:
        print(f"\nOpenAPI spec validation failed: {e}")
        raise

    # Create local spec file
    local_spec = tmp_path / "openapi.json"
    with local_spec.open("w") as f:
        json.dump(PETSTORE_SPEC, f)

    print(f"\nCreated local spec at: {local_spec}")
    return {
        "local_path": str(local_spec),
        "remote_url": "https://api.example.com/openapi.json",
    }


@pytest.mark.asyncio
async def test_openapi_toolset_local(mock_openapi_spec, caplog):
    """Test OpenAPI toolset with local file."""
    caplog.set_level("DEBUG")

    local_path = mock_openapi_spec["local_path"]
    toolset = OpenAPITools(
        spec=local_path,
        base_url="https://api.example.com",
    )

    # Load and validate spec
    spec = toolset._load_spec()
    print(f"\nLoaded spec: {spec}")
    validate(spec)
    print("Spec validation passed")

    # Store spec explicitly
    toolset._spec = spec
    toolset._operations = toolset._parse_operations()
    print(f"\nOperations: {toolset._operations}")

    # Get tools
    tools = toolset.get_llm_callable_tools()
    print(f"\nGenerated tools: {tools}")

    assert len(tools) == 1, f"Expected 1 tool, got {len(tools)}: {tools}"
    tool = tools[0]
    print("\nTool details:")
    print(f"Name: {tool.name}")
    print(f"Schema: {tool.get_schema()}")


@pytest.mark.asyncio
async def test_openapi_toolset_remote(mock_openapi_spec, caplog, monkeypatch):
    """Test OpenAPI toolset with remote spec."""
    caplog.set_level("DEBUG")

    url = mock_openapi_spec["remote_url"]

    # Create mock httpx response
    class MockResponse:
        status_code = 200

        @property
        def text(self):
            return json.dumps(PETSTORE_SPEC)

        def raise_for_status(self):
            pass

    # Mock httpx.get
    def mock_get(*args, **kwargs):
        return MockResponse()

    # Patch httpx.get
    monkeypatch.setattr("httpx.get", mock_get)

    toolset = OpenAPITools(
        spec=url,
        base_url="https://api.example.com",
    )

    # Load spec
    spec = toolset._load_spec()
    print(f"\nLoaded spec: {spec}")
    validate(spec)
    print("Spec validation passed")

    # Store spec explicitly
    toolset._spec = spec
    toolset._operations = toolset._parse_operations()
    print(f"\nOperations: {toolset._operations}")

    # Get tools
    tools = toolset.get_llm_callable_tools()
    print(f"\nGenerated tools: {tools}")

    assert len(tools) == 1, f"Expected 1 tool, got {len(tools)}: {tools}"
