# ruff: noqa: PLR2004

import json
from typing import Any
from uuid import UUID

import pytest
import upath
import yamling

from jinjarope.llm.apicallcollectionmanager import (
    APICall,
    APICallCollectionManager,
)


@pytest.fixture
def temp_collections_dir(tmp_path: upath.UPath) -> upath.UPath:
    collection_dir = tmp_path / "collections"
    collection_dir.mkdir()
    return collection_dir


@pytest.fixture
def sample_collection_data() -> dict[str, Any]:
    return {
        "name": "test_collection",
        "description": "Test collection",
        "context_bundles": [
            {
                "bundle_id": "750e8400-e29b-41d4-a716-446655440000",
                "name": "test_bundle",
                "sources": [
                    {
                        "source_type": "file",
                        "name": "bundle_file",
                        "file_path": "test.txt",
                    }
                ],
            }
        ],
        "calls": [
            {
                "call_id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "test_call_1",
                "prompts": [
                    {"prompt_type": "system", "content": "test prompt 1", "order": 0}
                ],
                "parameters": {"model": "test-model", "temperature": 0.7},
            }
        ],
    }


@pytest.fixture
def sample_collection_data_2() -> dict[str, Any]:
    return {
        "name": "test_collection_2",
        "description": "Test collection",
        "context_bundles": [
            {
                "bundle_id": "123e8400-e29b-41d4-a716-446655440000",
                "name": "test_bundle",
                "sources": [
                    {
                        "source_type": "file",
                        "name": "bundle_file",
                        "file_path": "test.txt",
                    }
                ],
            }
        ],
        "calls": [
            {
                "call_id": "12348400-e29b-41d4-a716-446655440000",
                "name": "test_call_2",
                "prompts": [
                    {"prompt_type": "system", "content": "test prompt 2", "order": 0}
                ],
                "parameters": {"model": "test-model", "temperature": 0.3},
            }
        ],
    }


@pytest.fixture
def collection_manager(temp_collections_dir: upath.UPath) -> APICallCollectionManager:
    return APICallCollectionManager([temp_collections_dir])


@pytest.fixture
def populated_manager(
    collection_manager: APICallCollectionManager,
    temp_collections_dir: upath.UPath,
    sample_collection_data: dict[str, Any],
    sample_collection_data_2: dict[str, Any],
) -> APICallCollectionManager:
    # Create test files in both YAML and JSON format
    (temp_collections_dir / "test1.yaml").write_text(
        yamling.dump_yaml(sample_collection_data)
    )
    (temp_collections_dir / "test2.json").write_text(json.dumps(sample_collection_data_2))
    collection_manager.load_collections([temp_collections_dir])
    return collection_manager


def test_load_collections(populated_manager: APICallCollectionManager):
    assert len(populated_manager.collections) == 2
    assert "test_collection" in populated_manager.collections


def test_get_collection(populated_manager: APICallCollectionManager):
    collection = populated_manager.get_collection("test_collection")
    assert collection is not None
    assert collection.name == "test_collection"
    assert len(collection.calls) == 1


def test_get_call(populated_manager: APICallCollectionManager):
    call_id = UUID("550e8400-e29b-41d4-a716-446655440000")
    call = populated_manager.get_call(call_id)
    assert call is not None
    assert call.name == "test_call_1"


def test_contains(populated_manager: APICallCollectionManager):
    call_id = UUID("550e8400-e29b-41d4-a716-446655440000")
    assert call_id in populated_manager
    assert "test_call_1" in populated_manager
    assert UUID("999e8400-e29b-41d4-a716-446655440000") not in populated_manager


def test_len(populated_manager: APICallCollectionManager):
    assert len(populated_manager) == 2


def test_iter(populated_manager: APICallCollectionManager):
    calls = list(populated_manager)
    assert len(calls) == 2
    assert all(isinstance(call, APICall) for call in calls)


def test_load_nonexistent_directory():
    with pytest.raises(FileNotFoundError):
        APICallCollectionManager(["nonexistent"])


def test_list_collections(populated_manager: APICallCollectionManager):
    collections = populated_manager.list_collections()
    assert len(collections) == 2
    assert "test_collection" in collections


def test_get_bundle(populated_manager: APICallCollectionManager):
    bundle = populated_manager.get_bundle("test_bundle")
    assert bundle is not None
    assert bundle.name == "test_bundle"
    assert len(bundle.sources) == 1


def test_list_bundles(populated_manager: APICallCollectionManager):
    bundles = populated_manager.list_bundles()
    assert len(bundles) == 2
    assert ("test_collection", "test_bundle") in bundles


# def test_expanded_sources(populated_manager: APICallCollectionManager):
#     call = populated_manager.get_call("test_call_1")
#     assert call
#     expanded = call.get_expanded_sources(populated_manager)
#     assert len(expanded) == 1
#     assert isinstance(expanded[0], FileSource)
#     assert expanded[0].name == "bundle_file"


def test_get_call_by_name(populated_manager: APICallCollectionManager):
    assert populated_manager.get_call("xyz") is None


if __name__ == "__main__":
    pytest.main(["-v", __file__])
