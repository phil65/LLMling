"""Example of concurrent task execution."""

from __future__ import annotations

from llmling.config.manager import ConfigManager
from llmling.context import default_registry as context_registry
from llmling.llm.registry import default_registry as llm_registry
from llmling.processors.base import ProcessorConfig
from llmling.processors.registry import ProcessorRegistry
from llmling.task.concurrent import execute_concurrent
from llmling.task.executor import TaskExecutor
from llmling.task.manager import TaskManager


async def register_providers(config_manager: ConfigManager) -> None:
    """Register all providers from config including groups."""
    # First register all direct providers
    for provider_key in config_manager.config.llm_providers:
        # Register with the provider key (e.g., 'local-llama'), not the display name
        llm_registry["litellm"] = provider_key
        print(f"Registered provider {provider_key} with implementation litellm")

    # Then register provider groups
    for group_name, providers in config_manager.config.provider_groups.items():
        if providers:  # Ensure the group has at least one provider
            # Register group using first provider's implementation
            llm_registry["litellm"] = group_name
            print(f"Registered provider group {group_name} with implementation litellm")


async def main() -> None:
    # Initialize processor registry and register processors
    processor_registry = ProcessorRegistry()
    await processor_registry.startup()

    try:
        # Register test processors
        processor_configs = {
            "python_cleaner": ProcessorConfig(
                type="function",
                import_path="llmling.testing.processors.uppercase_text",
            ),
            "sanitize": ProcessorConfig(
                type="function",
                import_path="llmling.testing.processors.uppercase_text",
            ),
            "add_metadata": ProcessorConfig(
                type="template",
                template="""
                # Generated at: {{ now() }}
                # Source: {{ source }}
                # Version: {{ version }}

                {{ content }}
                """,
            ),
        }

        for name, config in processor_configs.items():
            processor_registry.register(name, config)
            print(f"Registered processor: {name}")

        # Load configuration
        print("\nLoading configuration...")
        config_manager = ConfigManager.load("src/llmling/resources/test.yml")

        # Register all providers including groups
        print("\nRegistering providers...")
        await register_providers(config_manager)

        # Create executor and manager
        executor = TaskExecutor(
            context_registry=context_registry,
            processor_registry=processor_registry,
            provider_registry=llm_registry,
        )
        manager = TaskManager(config_manager.config, executor)

        # Execute task with the local-llama provider
        print("\nExecuting task...")
        results = await execute_concurrent(
            manager,
            templates=["quick_review"],  # Using the template that uses local-llama
            max_concurrent=2,
        )

        # Print results
        for result in results:
            print("\nResult:")
            print("-" * 40)
            print(f"Model: {result.model}")
            print(f"Content: {result.content}")
            print("\nContext metadata:")
            for key, value in result.context_metadata.items():
                print(f"  {key}: {value}")
            print("\nCompletion metadata:")
            for key, value in result.completion_metadata.items():
                print(f"  {key}: {value}")

    except Exception as exc:
        print(f"Error: {exc}")
        raise

    finally:
        print("\nShutting down...")
        await processor_registry.shutdown()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
