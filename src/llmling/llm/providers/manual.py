from __future__ import annotations

from llmling.llm.base import LLMConfig, Message, MessageContent, ToolCall
from llmling.llm.providers.dummy import DummyProvider
from llmling.resources import help_texts


def create_manual_provider() -> DummyProvider:
    """Create a dummy provider configured as manual/helper with shocking bills."""
    config = LLMConfig(
        model="manual/helper",
        provider_name="manual",
        metadata={
            "delay": 0.5,
            "pricing": {
                "input_cost_per_token": 0.5,  # $500 per 1k input tokens
                "output_cost_per_token": 1.0,  # $1000 per 1k output tokens
                "tokens_per_char": 0.25,
            },
            "responses": {
                "default": {
                    "content": (
                        help_texts.GENERAL_HELP
                        + "\n\nJust kidding about those costs by the way - "
                        "I'm the manual provider showing you a simulated high bill. 😉\n"
                        "Install a real provider to get actual pricing!"
                    ),
                    "metadata": {"help_type": "general", "just_kidding": True},
                },
                "condition:has_image=true": {
                    "content": (
                        help_texts.VISION_HELP
                        + "\n\nPhew, good thing this is just a simulation - "
                        "that image would have cost you $1,337.42 to process! 🎭"
                    ),
                    "metadata": {"help_type": "vision", "just_kidding": True},
                },
                "condition:has_tools=true": {
                    "content": (
                        help_texts.TOOLS_HELP
                        + "\n\nBy the way, if this were a real provider, "
                        "that tool call would have cost you $420.69. "
                        "Good thing I'm just here to help! 🎪"
                    ),
                    "metadata": {"help_type": "tools", "just_kidding": True},
                },
            },
        },
    )
    return DummyProvider(config)


if __name__ == "__main__":
    import asyncio

    async def demo():
        provider = create_manual_provider()

        # Test normal help
        print("\nBasic help:")
        result = await provider.complete([Message(role="user", content="Hello!")])
        print(f"Response: {result.content}")
        print(f"Your simulated bill: ${result.total_cost:.2f}")

        # Test vision help
        print("\nVision help:")
        result = await provider.complete([
            Message(
                role="user",
                content="What's in this image?",
                content_items=[
                    MessageContent(type="image_url", content="http://example.com/img.jpg")
                ],
            )
        ])
        print(f"Response: {result.content}")
        print(f"Your simulated bill: ${result.total_cost:.2f}")

        # Test tools help
        print("\nTools help:")
        result = await provider.complete([
            Message(
                role="user",
                content="Calculate this",
                tool_calls=[ToolCall(id="1", name="calculator", parameters={"x": 1})],
            )
        ])
        print(f"Response: {result.content}")
        print(f"Your simulated bill: ${result.total_cost:.2f}")

    asyncio.run(demo())
