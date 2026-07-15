"""Data schemas for exchange objects used throughout the library."""

from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Represents a document loaded from a source."""
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    id: str | None = None


class Chunk(BaseModel):
    """Represents a split chunk of a document."""
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    id: str | None = None
    document_id: str | None = None
    embedding: list[float] | None = None


class Filter(BaseModel):
    """Represents filtering criteria for vector searches."""
    metadata_filter: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Represents a result returned from vector store search."""
    chunk: Chunk
    score: float
