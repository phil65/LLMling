"""Model definitions for LLM capabilities and configurations.

This module provides Pydantic models for working with LLM capabilities,
particularly focused on litellm.get_model_info() output.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


OpenAIParam = Literal[
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "max_tokens",
    "max_completion_tokens",
    "modalities",
    "prediction",
    "n",
    "presence_penalty",
    "seed",
    "stop",
    "stream",
    "stream_options",
    "temperature",
    "top_p",
    "tools",
    "response_format",
    "tool_choice",
    "function_call",
    "functions",
    "max_retries",
    "extra_headers",
    "parallel_tool_calls",
    "user",
]

Mode = Literal["chat", "completion"]

Provider = Literal[
    "openai", "anthropic", "google", "bedrock", "ollama", "azure", "cohere"
]


class Capabilities(BaseModel):
    """Complete model capabilities from litellm.get_model_info()."""

    model_config = ConfigDict(extra="allow", frozen=True, populate_by_name=True)

    # Core model information
    key: str | None = None
    """Model identifier"""

    litellm_provider: Provider | None = None
    """Provider (e.g., 'ollama', 'openai')"""

    mode: Mode | None = None
    """Operation mode ('chat' or 'completion')"""

    # Token limits
    max_tokens: int | None = None
    """Maximum total tokens"""

    max_input_tokens: int | None = None
    """Maximum input tokens"""

    max_output_tokens: int | None = None
    """Maximum output tokens"""

    # Cost information
    input_cost_per_token: float | None = None
    """Cost per input token"""

    output_cost_per_token: float | None = None
    """Cost per output token"""

    cache_creation_input_token_cost: float | None = None
    """Cost per token for cache creation"""

    cache_read_input_token_cost: float | None = None
    """Cost per token for cache reading"""

    input_cost_per_character: float | None = None
    """Cost per character for input"""

    input_cost_per_token_above_128k_tokens: float | None = None
    """Cost per token for input above 128k tokens"""

    input_cost_per_query: float | None = None
    """Cost per query"""

    input_cost_per_second: float | None = None
    """Cost per second for input"""

    input_cost_per_audio_token: float | None = None
    """Cost per audio token for input"""

    input_cost_per_character_above_128k_tokens: float | None = None
    """Cost per character for input above 128k tokens"""

    input_cost_per_image: float | None = None
    """Cost per image for input"""

    input_cost_per_audio_per_second: float | None = None
    """Cost per second for audio input"""

    input_cost_per_video_per_second: float | None = None
    """Cost per second for video input"""

    output_cost_per_video_per_second: float | None = None
    """Cost per second for video output"""

    output_cost_per_audio_per_second: float | None = None
    """Cost per second for audio output"""

    output_cost_per_audio_token: float | None = None
    """Cost per audio token for output"""

    output_cost_per_character: float | None = None
    """Cost per character for output"""

    output_cost_per_token_above_128k_tokens: float | None = None
    """Cost per token for output above 128k tokens"""

    output_cost_per_character_above_128k_tokens: float | None = None
    """Cost per character for output above 128k tokens"""

    output_cost_per_second: float | None = None
    """Cost per second for output"""

    output_cost_per_image: float | None = None
    """Cost per image for output"""

    output_vector_size: int | None = None
    """Size of output vectors"""

    # OpenAI compatibility (Literals for LSP help)
    supported_openai_params: list[OpenAIParam | str] = Field(default_factory=list)
    """List of supported OpenAI parameters"""

    supports_function_calling: bool | None = None
    """Whether function calling is supported"""

    # Additional capabilities
    supports_system_messages: bool | None = None
    """Whether system messages are supported"""

    supports_response_schema: bool | None = None
    """Whether response schemas are supported"""

    supports_vision: bool | None = None
    """Whether vision/image input is supported"""

    supports_assistant_prefill: bool | None = None
    """Whether assistant prefill is supported"""

    supports_prompt_caching: bool | None = None
    """Whether prompt caching is supported"""

    supports_audio_input: bool | None = None
    """Whether audio input is supported"""

    supports_audio_output: bool | None = None
    """Whether audio output is supported"""

    @property
    def is_vision_capable(self) -> bool:
        """Check if the model has vision capabilities.

        First checks the supports_vision field from litellm info.
        If that's None or False, falls back to checking against known vision models.
        """
        # First check official support from litellm
        if self.supports_vision is True:
            return True

        # Fallback to known vision models map
        if not self.key:
            return False

        vision_models = {
            "openai": ["gpt-4-vision-preview"],
            "anthropic": [
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240229",
            ],
            "google": ["gemini-pro-vision"],
            "bedrock": [
                "anthropic.claude-3-sonnet-20240229-v1:0",
                "anthropic.claude-3-haiku-20240229-v1:0",
            ],
            "ollama": ["llava", "bakllava", "llava-13b", "llava-7b"],
        }

        return any(self.key in models for models in vision_models.values())


if __name__ == "__main__":
    import devtools
    import litellm

    info = litellm.get_model_info("gpt-4-vision-preview")
    print(devtools.debug(info))
