"""Module for managing context sources."""

from __future__ import annotations

from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl


class SourceType(str, Enum):
    FILE = "file"
    """Local file system source"""

    DATABASE = "database"
    """Database query source"""

    API = "api"
    """External API endpoint source"""

    MEMORY = "memory"
    """In-memory cached source"""

    BUNDLE = "bundle"
    """Reference to a context bundle"""


class ContextSource(BaseModel):
    source_id: UUID = Field(default_factory=uuid4)
    """Unique identifier for the context source"""

    source_type: SourceType
    """Type of the context source (file/db/api/memory)"""

    name: str
    """Human readable name for the context source"""

    description: str | None = None
    """Optional description of what this source provides"""


class FileSource(ContextSource):
    file_path: str
    """Path to the source file, absolute or relative"""

    line_start: int | None = None
    """Starting line number for partial file reads"""

    line_end: int | None = None
    """Ending line number for partial file reads"""


class DatabaseSource(ContextSource):
    connection_string: str
    """Database connection string with credentials"""

    query: str
    """SQL query to execute for retrieving context"""


class ApiSource(ContextSource):
    url: HttpUrl
    """API endpoint URL"""

    headers: dict[str, str] | None = None
    """Optional HTTP headers for the API request"""

    method: str = "GET"
    """HTTP method to use (GET/POST/etc)"""


class BundleSource(ContextSource):
    """References a context bundle to include its sources."""

    bundle_name: str
    """Name of the bundle to reference"""

    source_type: SourceType = SourceType.BUNDLE
    """Always BUNDLE for this source type"""
