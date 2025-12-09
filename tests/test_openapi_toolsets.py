from __future__ import annotations

import json
from typing import TYPE_CHECKING

from openapi_spec_validator import validate
from openapi_spec_validator.exceptions import OpenAPISpecValidatorError
import pytest

from llmling.tools.openapi import OpenAPITools


if TYPE_CHECKING:
    from jsonschema_path.typing import Schema


BASE_URL = "https://api.example.com"
PETSTORE_SPEC: Schema = {
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
    "servers": [{"url": BASE_URL, "description": "Pet store API server"}],
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
    local_spec.write_text(json.dumps(PETSTORE_SPEC))
    print(f"\nCreated local spec at: {local_spec}")
    return {
        "local_path": str(local_spec),
        "remote_url": f"{BASE_URL}/openapi.json",
    }


async def test_openapi_toolset_local(mock_openapi_spec):
    """Test OpenAPI toolset with local file."""
    local_path = mock_openapi_spec["local_path"]
    toolset = OpenAPITools(spec=local_path, base_url=BASE_URL)

    # Load spec (now sets internal state)
    toolset._load_spec()

    # Validate the loaded spec
    validate(toolset._spec)

    # Get tools
    tools = toolset.get_llm_callable_tools()

    assert len(tools) == 1, f"Expected 1 tool, got {len(tools)}: {tools}"


async def test_openapi_toolset_remote(mock_openapi_spec, caplog, monkeypatch):
    """Test OpenAPI toolset with remote spec."""
    caplog.set_level("DEBUG")

    url = mock_openapi_spec["remote_url"]

    # Mock load_openapi_spec to return local spec
    def mock_load_spec(spec_url):
        return PETSTORE_SPEC

    monkeypatch.setattr("schemez.openapi.load_openapi_spec", mock_load_spec)

    toolset = OpenAPITools(spec=url, base_url=BASE_URL)

    # Load spec (now sets internal state)
    toolset._load_spec()

    # Validate the loaded spec
    validate(toolset._spec)

    # Get tools
    tools = toolset.get_llm_callable_tools()
    assert len(tools) == 1, f"Expected 1 tool, got {len(tools)}: {tools}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
