"""Qdrant vector store implementation."""

import uuid
from typing import Any

from rag_qdrant.exceptions import CollectionNotFoundError, VectorStoreError
from rag_qdrant.interfaces.schemas import Chunk, Filter, SearchResult
from rag_qdrant.vectorstore.base import BaseVectorStore


class QdrantVectorStore(BaseVectorStore):
    """Concrete implementation of BaseVectorStore using Qdrant."""

    def __init__(
        self,
        location: str | None = None,
        url: str | None = None,
        port: int | None = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: bool | None = None,
        api_key: str | None = None,
        prefix: str | None = None,
        timeout: float | None = None,
        host: str | None = None,
        client: Any | None = None,
        aclient: Any | None = None,
        **kwargs: Any
    ) -> None:
        """Initialize the QdrantVectorStore.

        If client/aclient are provided, they are used directly. Otherwise, they are lazily
        instantiated using the connection parameters.
        """
        self.connection_args = {
            "location": location,
            "url": url,
            "port": port,
            "grpc_port": grpc_port,
            "prefer_grpc": prefer_grpc,
            "https": https,
            "api_key": api_key,
            "prefix": prefix,
            "timeout": timeout,
            "host": host,
            **kwargs
        }
        self._client = client
        self._aclient = aclient

    @property
    def client(self) -> Any:
        """Get the sync Qdrant client."""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as e:
                raise VectorStoreError(
                    "The 'qdrant-client' package is required. "
                    "Install it with: pip install qdrant-client"
                ) from e
            try:
                self._client = QdrantClient(**self.connection_args)
            except Exception as e:
                raise VectorStoreError(f"Failed to connect to Qdrant sync client: {e}") from e
        return self._client

    @property
    def aclient(self) -> Any:
        """Get the async Qdrant client."""
        if self._aclient is None:
            try:
                from qdrant_client import AsyncQdrantClient
            except ImportError as e:
                raise VectorStoreError(
                    "The 'qdrant-client' package is required. "
                    "Install it with: pip install qdrant-client"
                ) from e
            try:
                # Async client connection arguments might need minor adjustments
                # (e.g. prefer_grpc is not supported in the same way in all versions,
                # but keep it standard)
                self._aclient = AsyncQdrantClient(**self.connection_args)
            except Exception as e:
                raise VectorStoreError(f"Failed to connect to Qdrant async client: {e}") from e
        return self._aclient

    def _resolve_distance(self, distance_str: str) -> Any:
        """Resolve a string distance metric name to Qdrant models."""
        from qdrant_client.http import models as rest_models
        mapping = {
            "cosine": rest_models.Distance.COSINE,
            "dot": rest_models.Distance.DOT,
            "euclid": rest_models.Distance.EUCLID,
        }
        dist = mapping.get(distance_str.lower())
        if dist is None:
            raise VectorStoreError(f"Unsupported distance metric: '{distance_str}'")
        return dist

    def _parse_filter(self, filters: Filter | None) -> Any | None:
        """Convert domain Filter schema to Qdrant models."""
        if not filters or not filters.metadata_filter:
            return None
        from qdrant_client.http import models as rest_models
        
        conditions: list[Any] = []
        for key, value in filters.metadata_filter.items():
            # If the value is a list, match any of the values
            if isinstance(value, list):
                conditions.append(
                    rest_models.FieldCondition(
                        key=f"metadata.{key}",
                        match=rest_models.MatchAny(any=value)
                    )
                )
            else:
                conditions.append(
                    rest_models.FieldCondition(
                        key=f"metadata.{key}",
                        match=rest_models.MatchValue(value=value)
                    )
                )
        return rest_models.Filter(must=conditions)

    def create_collection(
        self,
        collection_name: str,
        vector_size: int = 1024,
        distance: str = "Cosine"
    ) -> None:
        """Create a collection in Qdrant."""
        from qdrant_client.http import models as rest_models
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=rest_models.VectorParams(
                    size=vector_size,
                    distance=self._resolve_distance(distance)
                )
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to create collection '{collection_name}': {e}") from e

    def upsert(self, collection_name: str, chunks: list[Chunk]) -> None:
        """Upsert points into a Qdrant collection."""
        from qdrant_client.models import PointStruct
        
        # Verify collection exists first
        if not self.client.collection_exists(collection_name):
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found.")

        points = []
        for chunk in chunks:
            if not chunk.embedding:
                raise VectorStoreError(f"Chunk with ID {chunk.id} has no embedding. Cannot upsert.")
            
            point_id: str | int = chunk.id or str(uuid.uuid4())
            if isinstance(point_id, str) and point_id.isdigit():
                point_id = int(point_id)
            payload = {
                "content": chunk.content,
                "document_id": chunk.document_id,
                "metadata": chunk.metadata,
            }
            points.append(
                PointStruct(
                    id=point_id,
                    vector=chunk.embedding,
                    payload=payload
                )
            )

        try:
            self.client.upsert(collection_name=collection_name, points=points)
        except Exception as e:
            raise VectorStoreError(
                f"Failed to upsert points into collection '{collection_name}': {e}"
            ) from e

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        filters: Filter | None = None
    ) -> list[SearchResult]:
        """Perform similarity search on Qdrant."""
        if not self.client.collection_exists(collection_name):
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found.")

        qdrant_filter = self._parse_filter(filters)
        
        try:
            response = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=limit
            )
            hits = response.points
            
            results = []
            for hit in hits:
                payload = hit.payload or {}
                chunk = Chunk(
                    content=payload.get("content", ""),
                    document_id=payload.get("document_id"),
                    metadata=payload.get("metadata", {}),
                    id=str(hit.id)
                )
                results.append(
                    SearchResult(
                        chunk=chunk,
                        score=hit.score
                    )
                )
            return results
        except Exception as e:
            raise VectorStoreError(f"Search failed in collection '{collection_name}': {e}") from e

    def delete(self, collection_name: str, filters: Filter | None = None) -> None:
        """Delete items from collection."""
        if not self.client.collection_exists(collection_name):
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found.")
            
        qdrant_filter = self._parse_filter(filters)
        if qdrant_filter is None:
            # Avoid accidental deletion of entire database
            raise VectorStoreError(
                "Delete operation requires a valid filter to prevent total data loss."
            )
            
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=qdrant_filter
            )
        except Exception as e:
            raise VectorStoreError(
                f"Delete operation failed in collection '{collection_name}': {e}"
            ) from e

    # Async Implementations

    async def acreate_collection(
        self,
        collection_name: str,
        vector_size: int = 1024,
        distance: str = "Cosine"
    ) -> None:
        """Create a collection asynchronously."""
        from qdrant_client.http import models as rest_models
        try:
            await self.aclient.create_collection(
                collection_name=collection_name,
                vectors_config=rest_models.VectorParams(
                    size=vector_size,
                    distance=self._resolve_distance(distance)
                )
            )
        except Exception as e:
            raise VectorStoreError(
                f"Failed to create collection '{collection_name}' asynchronously: {e}"
            ) from e

    async def aupsert(self, collection_name: str, chunks: list[Chunk]) -> None:
        """Upsert points asynchronously."""
        from qdrant_client.models import PointStruct
        
        # Verify collection exists
        if not await self.aclient.collection_exists(collection_name):
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found.")

        points = []
        for chunk in chunks:
            if not chunk.embedding:
                raise VectorStoreError(f"Chunk with ID {chunk.id} has no embedding. Cannot upsert.")
            
            point_id: str | int = chunk.id or str(uuid.uuid4())
            if isinstance(point_id, str) and point_id.isdigit():
                point_id = int(point_id)
            payload = {
                "content": chunk.content,
                "document_id": chunk.document_id,
                "metadata": chunk.metadata,
            }
            points.append(
                PointStruct(
                    id=point_id,
                    vector=chunk.embedding,
                    payload=payload
                )
            )

        try:
            await self.aclient.upsert(collection_name=collection_name, points=points)
        except Exception as e:
            raise VectorStoreError(f"Failed to upsert points asynchronously: {e}") from e

    async def asearch(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        filters: Filter | None = None
    ) -> list[SearchResult]:
        """Perform similarity search asynchronously."""
        if not await self.aclient.collection_exists(collection_name):
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found.")

        qdrant_filter = self._parse_filter(filters)
        
        try:
            response = await self.aclient.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=limit
            )
            hits = response.points
            
            results = []
            for hit in hits:
                payload = hit.payload or {}
                chunk = Chunk(
                    content=payload.get("content", ""),
                    document_id=payload.get("document_id"),
                    metadata=payload.get("metadata", {}),
                    id=str(hit.id)
                )
                results.append(
                    SearchResult(
                        chunk=chunk,
                        score=hit.score
                    )
                )
            return results
        except Exception as e:
            raise VectorStoreError(
                f"Async search failed in collection '{collection_name}': {e}"
            ) from e

    async def adelete(self, collection_name: str, filters: Filter | None = None) -> None:
        """Delete items asynchronously."""
        if not await self.aclient.collection_exists(collection_name):
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found.")
            
        qdrant_filter = self._parse_filter(filters)
        if qdrant_filter is None:
            raise VectorStoreError(
                "Delete operation requires a valid filter to prevent total data loss."
            )
            
        try:
            await self.aclient.delete(
                collection_name=collection_name,
                points_selector=qdrant_filter
            )
        except Exception as e:
            raise VectorStoreError(
                f"Async delete operation failed in collection '{collection_name}': {e}"
            ) from e

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection from Qdrant."""
        try:
            self.client.delete_collection(collection_name=collection_name)
        except Exception as e:
            raise VectorStoreError(f"Failed to delete collection '{collection_name}': {e}") from e

    async def adelete_collection(self, collection_name: str) -> None:
        """Delete a collection asynchronously from Qdrant."""
        try:
            await self.aclient.delete_collection(collection_name=collection_name)
        except Exception as e:
            raise VectorStoreError(
                f"Failed to delete collection '{collection_name}' asynchronously: {e}"
            ) from e
