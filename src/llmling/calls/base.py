"""Base classes for API calls."""

from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from jinjarope.llm.sources import ContextSource


class APICall(BaseModel):
    call_id: UUID = Field(default_factory=uuid4)
    """Unique identifier for this API call"""

    name: str
    """Human readable name for the API call"""

    description: str | None = None
    """Optional description of the call's purpose"""

    context_sources: list[ContextSource] = Field(default_factory=list)
    """Optional list of context sources to include"""

    def execute(self) -> dict:
        """Execute the API call."""
        msg = "Subclasses should implement this method"
        raise NotImplementedError(msg)
