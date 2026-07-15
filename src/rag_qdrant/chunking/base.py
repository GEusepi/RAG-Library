"""Base interface for document chunkers."""

from abc import ABC, abstractmethod

from rag_qdrant.interfaces.schemas import Chunk, Document


class BaseChunker(ABC):
    """Abstract base class for all chunkers."""

    @abstractmethod
    def chunk(self, documents: list[Document]) -> list[Chunk]:
        """Split a list of documents into chunks synchronously.

        Args:
            documents: The list of documents to chunk.

        Returns:
            List[Chunk]: The list of created chunks.

        Raises:
            ChunkingError: If chunking fails.
        """
        pass

    @abstractmethod
    async def achunk(self, documents: list[Document]) -> list[Chunk]:
        """Split a list of documents into chunks asynchronously.

        Args:
            documents: The list of documents to chunk.

        Returns:
            List[Chunk]: The list of created chunks.

        Raises:
            ChunkingError: If chunking fails.
        """
        pass
