"""Manages collections of API calls from multiple sources."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from uuid import UUID

import upath
import yaml

from jinjarope.llm.apicallcollection import (
    APICall,
    APICallCollection,
    ContextBundle,
)


if TYPE_CHECKING:
    from collections.abc import Iterator
    import os


class APICallCollectionManager:
    """Manages multiple API call collections loaded from files."""

    def __init__(self, collection_paths: list[str | os.PathLike[str]] | None = None):
        """Initialize the collection manager.

        Args:
            collection_paths: Optional list of paths to load collections from
        """
        self.collections: dict[str, APICallCollection] = {}
        if collection_paths:
            self.load_collections(collection_paths)

    def __len__(self) -> int:
        """Return the total number of API calls across all collections."""
        return sum(len(collection.calls) for collection in self.collections.values())

    def __iter__(self) -> Iterator[APICall]:
        """Iterate over all API calls."""
        for collection in self.collections.values():
            yield from collection.calls

    def __getitem__(self, key: UUID | str) -> APICall:
        """Get a call by UUID or name.

        Args:
            key: UUID or name of the call to retrieve

        Returns:
            The matching API call

        Raises:
            KeyError: If no matching call is found
        """
        call = self.get_call(key)
        if call is None:
            msg = f"Call '{key}' not found"
            raise KeyError(msg)
        return call

    def __contains__(self, item: UUID | str | APICall) -> bool:
        """Check if a call exists using ID, name or call object."""
        match item:
            case UUID() | str():
                return self.get_call(item) is not None
            case APICall():
                return self.get_call(item.call_id) is not None
            case _:
                return False

    def __str__(self) -> str:
        """Return a human readable summary."""
        return f"APICallManager with {len(self)} calls"

    def __repr__(self) -> str:
        """Return a detailed representation."""
        calls = [f"{call.name}({call.call_id})" for call in self]
        return f"APICallManager(calls=[{', '.join(calls)}])"

    def load_collections(self, paths: list[os.PathLike[str] | str]) -> None:
        """Load API call collections from the specified paths.

        Args:
            paths: List of paths to load collections from. Can be directories or files.

        Raises:
            FileNotFoundError: If a specified path doesn't exist
        """
        for p in paths:
            path = upath.UPath(p)
            if not path.exists():
                msg = f"Path {path} not found"
                raise FileNotFoundError(msg)

            # files = path.glob("*.{yaml,yml,toml,json}") if path.is_dir() else [path]
            files = list(path.iterdir()) if path.is_dir() else [path]
            for file_path in files:
                self._load_collection_file(file_path)

    def _load_collection_file(self, file_path: str | os.PathLike[str]):
        """Load a single collection file.

        Args:
            file_path: Path to the collection file to load

        Raises:
            yaml.YAMLError: On YAML parsing errors
            json.JSONDecodeError: On JSON parsing errors
        """
        path = upath.UPath(file_path)
        with path.open() as f:
            if path.suffix.lower() in (".yaml", ".yml"):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
            collection = APICallCollection(**data)
            self.collections[collection.name] = collection

    def get_collection(self, name: str) -> APICallCollection | None:
        """Get a collection by name."""
        return self.collections.get(name)

    def get_call(self, identifier: UUID | str) -> APICall | None:
        """Get an API call by UUID or name.

        Args:
            identifier: Either a UUID or string name of the call to retrieve

        Returns:
            The matching API call or None if not found
        """
        return next(
            (
                call
                for collection in self.collections.values()
                for call in collection.calls
                if (isinstance(identifier, UUID) and call.call_id == identifier)
                or (isinstance(identifier, str) and call.name == identifier)
            ),
            None,
        )

    def get_bundle(self, identifier: UUID | str) -> ContextBundle | None:
        """Get a context bundle by UUID or name.

        Args:
            identifier: Either a UUID or string name of the bundle to retrieve

        Returns:
            The matching context bundle or None if not found
        """
        return next(
            (
                bundle
                for collection in self.collections.values()
                for bundle in collection.context_bundles
                if (isinstance(identifier, UUID) and bundle.bundle_id == identifier)
                or (isinstance(identifier, str) and bundle.name == identifier)
            ),
            None,
        )

    def list_bundles(self) -> list[tuple[str, str]]:
        """Get a list of all context bundles as (collection_name, bundle_name) tuples."""
        return [
            (coll_name, bundle.name)
            for coll_name, collection in self.collections.items()
            for bundle in collection.context_bundles
        ]

    def list_collections(self) -> list[str]:
        """Get a list of all collection names."""
        return list(self.collections.keys())


if __name__ == "__main__":
    manager = APICallCollectionManager([upath.UPath("src/jinjarope/llm/prompts")])
    print(manager.list_collections())
