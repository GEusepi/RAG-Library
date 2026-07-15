"""Base interface for document loaders."""

from abc import ABC, abstractmethod

from rag_qdrant.interfaces.schemas import Document


class BaseLoader(ABC):
    """Abstract base class for all document loaders."""

    @abstractmethod
    def load(self) -> list[Document]:
        """Load documents synchronously.

        Returns:
            List[Document]: A list of loaded Document objects.

        Raises:
            LoaderError: If loading fails.
        """
        pass

    @abstractmethod
    async def aload(self) -> list[Document]:
        """Load documents asynchronously.

        Returns:
            List[Document]: A list of loaded Document objects.

        Raises:
            LoaderError: If loading fails.
        """
        pass
