GENERAL_HELP = """
👋 Hello! I'm here to help you get started with LLMling!

To start using LLMling with real models, install a provider:
- For cloud services (OpenAI, Anthropic, etc):
  pip install llmling-provider-litellm
- For local models:
  pip install llmling-provider-llm

See all providers at: https://llmling.readthedocs.io/providers
"""

VISION_HELP = """
I notice you're trying to use vision capabilities!
To work with images, install a vision-capable provider:

pip install llmling-provider-litellm  # For GPT-4 Vision
pip install llmling-provider-llm      # For local vision models
"""

TOOLS_HELP = """
I see you're trying to use tools!
To use LLM tools, install a provider that supports function calling:

pip install llmling-provider-litellm  # For OpenAI function calling

See https://llmling.readthedocs.io/tools for more information
"""
