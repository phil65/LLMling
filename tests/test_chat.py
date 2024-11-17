import pytest

from llmling.chat import Chat


def test_chat_from_yaml():
    chat = Chat.from_yaml("/path/to/prompts/example_collection.yml")
    assert chat is not None
    assert len(chat._conversation) > 0
    assert len(chat._tool_registry) > 0

def test_chat_completion():
    chat = Chat.from_yaml("/path/to/prompts/example_collection.yml")
    result = chat.complete()
    assert result.content is not None


if __name__ == "__main__":
    pytest.main([__file__])
