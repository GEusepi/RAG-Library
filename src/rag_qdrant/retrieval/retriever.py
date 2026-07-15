"""Standard dense vector retriever implementation."""


from rag_qdrant.embeddings.base import BaseEmbedder
from rag_qdrant.interfaces.schemas import Filter, SearchResult
from rag_qdrant.vectorstore.base import BaseVectorStore


class Retriever:
    """Orchestrates similarity search using an embedder and a vector store."""

    def __init__(
        self,
        store: BaseVectorStore,
        embedder: BaseEmbedder,
        collection_name: str
    ) -> None:
        """Initialize the Retriever.

        Args:
            store: The vector store instance to search in.
            embedder: The embedder used to convert text queries to vectors.
            collection_name: Name of the collection to query.
        """
        self.store = store
        self.embedder = embedder
        self.collection_name = collection_name

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        filters: Filter | None = None
    ) -> list[SearchResult]:
        """Retrieve relevant document chunks for a query.

        Args:
            query: The user query text.
            limit: Maximum number of results to return.
            filters: Filters to apply.

        Returns:
            List[SearchResult]: Relevant search results.
        """
        query_vector = self.embedder.embed_query(query)
        return self.store.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            filters=filters
        )

    async def aretrieve(
        self,
        query: str,
        limit: int = 5,
        filters: Filter | None = None
    ) -> list[SearchResult]:
        """Retrieve relevant document chunks asynchronously.

        Args:
            query: The user query text.
            limit: Maximum number of results to return.
            filters: Filters to apply.

        Returns:
            List[SearchResult]: Relevant search results.
        """
        query_vector = await self.embedder.aembed_query(query)
        return await self.store.asearch(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            filters=filters
        )
