"""Module for executing API calls using the LiteLLM library."""

from __future__ import annotations

from typing import TYPE_CHECKING

import litellm
import upath

from jinjarope.llm.apicallcollection import (
    APICall,
    ApiSource,
    ContextSource,
    PromptType,
)
from jinjarope.llm.apicallcollectionmanager import APICallCollectionManager
from jinjarope.llm.sources import BundleSource, DatabaseSource, FileSource


if TYPE_CHECKING:
    from uuid import UUID

litellm.set_verbose = True


class LiteLLMExecutor:
    """Executes API calls using the LiteLLM library."""

    def __init__(
        self,
        collection_manager: APICallCollectionManager | None = None,
        fallback_models: list[str] | None = None,
        with_fallbacks: bool = False,
        max_retries: int = 3,
    ):
        """Initialize the executor.

        Args:
            collection_manager: Optional manager for resolving context bundles
            fallback_models: List of model names to try if primary model fails
            with_fallbacks: Whether to use fallback models on failure
            max_retries: Maximum number of retries when using fallback models
        """
        self.loaded_contexts: dict[str | UUID, str] = {}
        self.collection_manager = collection_manager
        self.fallback_models = fallback_models or []
        self.with_fallbacks = with_fallbacks
        self.max_retries = max_retries

    def set_fallback_options(self, with_fallbacks: bool, max_retries: int) -> None:
        """Set fallback options for the executor.

        Args:
            with_fallbacks: Whether to use fallback models on failure
            max_retries: Maximum number of retries when using fallback models
        """
        self.with_fallbacks = with_fallbacks
        self.max_retries = max_retries

    def _load_file_source(self, source: FileSource) -> str:
        """Load content from a file source."""
        path = upath.UPath(source.file_path)
        if not path.exists():
            msg = f"File {path} not found"
            raise FileNotFoundError(msg)

        with path.open() as f:
            content = f.read()
            if source.line_start is not None and source.line_end is not None:
                lines = content.splitlines()
                content = "\n".join(lines[source.line_start : source.line_end])
        return content

    def _load_api_source(self, source: ApiSource) -> str:
        """Load content from an API source."""
        import httpx

        response = httpx.request(
            method=source.method,
            url=str(source.url),
            headers=source.headers or {},
        )
        response.raise_for_status()
        return response.text

    def _load_db_source(self, source: DatabaseSource) -> str:
        """Load content from a database source."""
        import sqlalchemy

        engine = sqlalchemy.create_engine(source.connection_string)
        with engine.connect() as conn:
            result = conn.execute(sqlalchemy.text(source.query))
            return "\n".join(str(row) for row in result)

    def load_context(self, call: APICall) -> dict[str, str]:
        """Load all context sources for an API call."""
        contexts: dict[str, str] = {}
        sources_to_load: list[ContextSource] = []

        # First pass - collect all sources including from bundles
        for source in call.context_sources:
            if source.source_id in self.loaded_contexts:
                contexts[source.name] = self.loaded_contexts[source.source_id]
                continue

            if isinstance(source, BundleSource):
                if self.collection_manager is None:
                    continue
                bundle = self.collection_manager.get_bundle(source.bundle_name)
                if bundle:
                    sources_to_load.extend(bundle.sources)
            else:
                sources_to_load.append(source)

        # Second pass - load all collected sources
        for source in sources_to_load:
            if source.source_id in self.loaded_contexts:
                contexts[source.name] = self.loaded_contexts[source.source_id]
                continue

            content = ""
            match source:
                case FileSource():
                    content = self._load_file_source(source)
                case ApiSource():
                    content = self._load_api_source(source)
                case DatabaseSource():
                    content = self._load_db_source(source)
                case _:
                    continue

            self.loaded_contexts[source.source_id] = content
            contexts[source.name] = content

        return contexts

    def format_messages(
        self, call: CompletionCall, contexts: dict[str, str]
    ) -> list[dict[str, str]]:
        """Format prompts and contexts into LiteLLM messages."""
        messages: list[dict[str, str]] = []

        # Add contexts as system messages first
        for name, content in contexts.items():
            messages.append({
                "role": "system",
                "content": f"Context from {name}:\n{content}",
            })

        # Add prompts in order
        messages.extend([
            prompt.to_message_content()
            for prompt in sorted(call.prompts, key=lambda p: p.order)
        ])

        return messages

    def execute(self, call: CompletionCall) -> str:
        """Execute an API call using LiteLLM, with optional fallback models."""
        contexts = self.load_context(call)
        messages = self.format_messages(call, contexts)
        params = call.parameters.model_dump(exclude={"tools", "tool_choice"})
        model = params.pop("model")

        for attempt in range(self.max_retries if self.with_fallbacks else 1):
            try:
                response = litellm.completion(
                    model=model,
                    messages=messages,
                    **params,
                )
                return response.choices[0].message.content
            except Exception:
                if (
                    self.with_fallbacks
                    and attempt < self.max_retries - 1
                    and self.fallback_models
                ):
                    model = self.fallback_models[attempt % len(self.fallback_models)]
                else:
                    raise
        msg = "LiteLLM execution failed"
        raise RuntimeError(msg)

    def generate_image(
        self,
        prompt: str,
        model: str = "dall-e-3",
        size: str = "1024x1024",
        quality: str = "standard",
        n: int = 1,
    ) -> list[str]:
        """Synchronous version of generate_image."""
        response = litellm.image_generation(
            prompt=prompt,
            model=model,
            size=size,
            quality=quality,
            n=n,
        )
        return [img.url for img in response.data]

    def get_available_models(self) -> list[str]:
        """Get list of available models."""
        return litellm.get_model_list()

    def calculate_cost(self, response: litellm.ModelResponse) -> float:
        """Calculate cost for a completed API call."""
        return litellm.completion_cost(completion_response=response)


if __name__ == "__main__":
    from jinjarope.llm.calls.completion import (
        CompletionCall,
        CompletionParameters,
        TextContent,
    )
    from jinjarope.llm.prompts import PromptType

    # Create a sample API call
    sample_call = CompletionCall(
        name="Test Chat",
        description="A test chat completion",
        prompts=[
            TextContent(
                type=PromptType.SYSTEM, text="You are a helpful assistant.", order=0
            ),
            TextContent(
                type=PromptType.USER, text="What is the capital of France?", order=1
            ),
        ],
        parameters=CompletionParameters(
            model="openai/gpt-3.5-turbo",
            temperature=0.7,
            response_format="json",
        ),
    )

    # Initialize the executor
    manager = APICallCollectionManager([upath.UPath("src/jinjarope/llm/prompts")])
    executor = LiteLLMExecutor(
        collection_manager=manager,
        fallback_models=["openai/gpt-3.5-turbo", "gemini/gemini-1.5-flash"],
    )

    # Set fallback options
    executor.set_fallback_options(with_fallbacks=True, max_retries=3)

    # Execute call
    try:
        result = executor.execute(sample_call)
        print("Response:", result)
    except Exception as e:
        print(f"Error: {e}")
