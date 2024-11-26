"""Tests for server session management."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from llmling.server.session import SessionState


if TYPE_CHECKING:
    from llmling.server import LLMLingServer


@pytest.mark.asyncio
async def test_session_state_transitions(
    server: LLMLingServer,
):
    """Test session state transitions."""
    session = server.session

    # Initial state
    assert session.state == SessionState.INITIALIZING

    # Start session
    await session.startup()
    assert session.state == SessionState.RUNNING

    # Close session
    await session.close()
    assert session.state == SessionState.CLOSED


@pytest.mark.asyncio
async def test_invalid_state_transitions(server: LLMLingServer):
    """Test invalid session state transitions are caught."""
    session = server.session

    # Can't go from INITIALIZING to CLOSED
    with pytest.raises(ValueError, match="Invalid state transition"):
        session.state = SessionState.CLOSED

    # Start normally
    await session.startup()
    assert session.state == SessionState.RUNNING

    # Can't go back to INITIALIZING
    with pytest.raises(ValueError, match="Invalid state transition"):
        session.state = SessionState.INITIALIZING


# @pytest.mark.asyncio
# async def test_message_handling(server: LLMLingServer):
#     """Test session message handling."""
#     session = server.session
#     await session.startup()

#     # Valid message
#     test_msg = {"type": "text", "content": "test"}
#     await session.send_message(test_msg)

#     async with session.process_messages() as messages:
#         async for msg in messages:
#             assert msg == test_msg
#             break

#     await session.close()
