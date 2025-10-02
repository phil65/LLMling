"""Resource loader for Git repositories."""

from __future__ import annotations

import tempfile
from time import time
from typing import TYPE_CHECKING, ClassVar

from llmling.config.models import RepositoryResource
from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.resources.base import ResourceLoader, create_loaded_resource
from llmling.utils.paths import guess_mime_type


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    import os

    from git.repo import Repo
    from upath.types import JoinablePath

    from llmling.processors.registry import ProcessorRegistry
    from llmling.resources.base import LoaderContext
    from llmling.resources.models import LoadedResource


logger = get_logger(__name__)


class RepositoryResourceLoader(ResourceLoader[RepositoryResource]):
    """Loads content from Git repositories."""

    context_class = RepositoryResource
    uri_scheme = "repository"
    supported_mime_types: ClassVar[list[str]] = ["text/plain", "text/x-python"]

    def __init__(self, context: LoaderContext[RepositoryResource] | None = None) -> None:
        super().__init__(context)
        self._cache: dict[str, tuple[Repo, float]] = {}
        self._cache_timeout = 300  # 5 minutes

    def _get_cached_repo(self, repo_url: str) -> Repo | None:
        """Get cached repository if not expired."""
        if cached := self._cache.get(repo_url):
            repo, last_access = cached
            if time() - last_access <= self._cache_timeout:
                self._cache[repo_url] = (repo, time())  # Update access time
                return repo
            # Remove expired cache entry
            self._cache.pop(repo_url)
        return None

    def _cache_repo(self, repo_url: str, repo: Repo) -> None:
        """Cache repository for reuse."""
        self._cache[repo_url] = (repo, time())

    def _create_resource(
        self,
        path: str | os.PathLike[str] | JoinablePath,
        name: str,
        resource: RepositoryResource,
    ) -> LoadedResource:
        """Create LoadedResource from file."""
        from upathtools import to_upath

        path_obj = to_upath(path)
        try:
            content = path_obj.read_text("utf-8")
            description = f"Repository content from {resource.repo_url} ({resource.ref})"
            return create_loaded_resource(
                content=content,
                source_type="repository",
                uri=self.create_uri(name=name),
                mime_type=guess_mime_type(path),
                name=resource.description or path_obj.name,
                description=description,
                additional_metadata={
                    "repo": resource.repo_url,
                    "ref": resource.ref,
                    "path": str(path),
                },
            )
        except Exception as exc:
            msg = f"Failed to create resource from {path}: {exc}"
            raise exceptions.LoaderError(msg) from exc

    async def _load_impl(
        self,
        resource: RepositoryResource,
        name: str,
        processor_registry: ProcessorRegistry | None,
    ) -> AsyncIterator[LoadedResource]:
        """Load git content."""
        import git
        from upathtools import to_upath

        try:
            repo = self._get_cached_repo(resource.repo_url)
            if not repo:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    logger.debug("Cloning %s to %s", resource.repo_url, tmp_dir)
                    repo = git.Repo.clone_from(
                        resource.repo_url,
                        tmp_dir,
                        branch=resource.ref,
                    )
                    if resource.sparse_checkout:
                        path_str = " ".join(resource.sparse_checkout)
                        repo.git.sparse_checkout("set", path_str)
                    self._cache_repo(resource.repo_url, repo)

            # Switch to requested ref
            repo.git.checkout(resource.ref)

            base_path = to_upath(repo.working_dir) / resource.path.lstrip("/")
            if base_path.is_file():
                loaded = self._create_resource(base_path, name, resource)
                if processor_registry and resource.processors:
                    result = await processor_registry.process(
                        loaded.content, resource.processors
                    )
                    loaded.content = result.content
                yield loaded
            else:
                # Directory - yield all matching files
                files = [p for p in base_path.rglob("*") if p.is_file()]
                for file_path in files:
                    rel_path = file_path.relative_to(base_path)
                    loaded = self._create_resource(file_path, str(rel_path), resource)
                    if processor_registry and resource.processors:
                        result = await processor_registry.process(
                            loaded.content, resource.processors
                        )
                        loaded.content = result.content
                    yield loaded

        except git.exc.GitCommandError as exc:  # type: ignore
            msg = f"Git operation failed: {exc}"
            raise exceptions.LoaderError(msg) from exc
        except Exception as exc:
            msg = f"Failed to load repository content: {exc}"
            raise exceptions.LoaderError(msg) from exc
