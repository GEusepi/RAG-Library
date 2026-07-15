"""Base interface for vector stores."""

from abc import ABC, abstractmethod

from rag_qdrant.interfaces.schemas import Chunk, Filter, SearchResult


class BaseVectorStore(ABC):
    """Abstract base class for all vector database integrations."""

    @abstractmethod
    def create_collection(
        self,
        collection_name: str,
        vector_size: int = 1024,
        distance: str = "Cosine"
    ) -> None:
        """Create a collection/index in the vector store.

        Args:
            collection_name: Name of the collection.
            vector_size: Dimensionality of the vectors.
            distance: Distance metric (e.g., 'Cosine', 'Euclid', 'Dot').
        """
        pass

    @abstractmethod
    def upsert(self, collection_name: str, chunks: list[Chunk]) -> None:
        """Insert or update chunks (with embeddings) in the vector store.

        Args:
            collection_name: Target collection name.
            chunks: List of Chunk objects containing embeddings.
        """
        pass

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        filters: Filter | None = None
    ) -> list[SearchResult]:
        """Perform a vector similarity search.

        Args:
            collection_name: Collection name to search.
            query_vector: The query embedding.
            limit: Maximum number of results to return.
            filters: Filtering criteria to apply.

        Returns:
            List[SearchResult]: Similar documents sorted by score descending.
        """
        pass

    @abstractmethod
    def delete(self, collection_name: str, filters: Filter | None = None) -> None:
        """Delete entries from the vector store matching the filter.

        Args:
            collection_name: Collection name.
            filters: Filter matching the records to delete.
        """
        pass

    # Async variants

    @abstractmethod
    async def acreate_collection(
        self,
        collection_name: str,
        vector_size: int = 1024,
        distance: str = "Cosine"
    ) -> None:
        """Create a collection asynchronously.

        Args:
            collection_name: Name of the collection.
            vector_size: Dimensionality of the vectors.
            distance: Distance metric.
        """
        pass

    @abstractmethod
    async def aupsert(self, collection_name: str, chunks: list[Chunk]) -> None:
        """Insert or update chunks asynchronously.

        Args:
            collection_name: Target collection name.
            chunks: List of Chunk objects containing embeddings.
        """
        pass

    @abstractmethod
    async def asearch(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        filters: Filter | None = None
    ) -> list[SearchResult]:
        """Perform a vector similarity search asynchronously.

        Args:
            collection_name: Collection name to search.
            query_vector: The query embedding.
            limit: Maximum number of results to return.
            filters: Filtering criteria.

        Returns:
            List[SearchResult]: Similar documents.
        """
        pass

    @abstractmethod
    async def adelete(self, collection_name: str, filters: Filter | None = None) -> None:
        """Delete entries asynchronously.

        Args:
            collection_name: Collection name.
            filters: Filter matching the records to delete.
        """
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """Delete an entire collection/index from the vector store.

        Args:
            collection_name: Name of the collection to delete.
        """
        pass

    @abstractmethod
    async def adelete_collection(self, collection_name: str) -> None:
        """Delete an entire collection/index asynchronously from the vector store.

        Args:
            collection_name: Name of the collection to delete.
        """
        pass
