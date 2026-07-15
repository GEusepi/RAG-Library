"""Hybrid retriever combining dense semantic search and keyword search."""


from rag_qdrant.embeddings.base import BaseEmbedder
from rag_qdrant.interfaces.schemas import Chunk, Filter, SearchResult
from rag_qdrant.vectorstore.qdrant import QdrantVectorStore


class HybridRetriever:
    """Retriever that merges semantic search and full-text keyword search using RRF."""

    def __init__(
        self,
        store: QdrantVectorStore,
        embedder: BaseEmbedder,
        collection_name: str,
        rrf_k: int = 60
    ) -> None:
        """Initialize the HybridRetriever.

        Args:
            store: Qdrant vector store.
            embedder: The embedder for dense query vectors.
            collection_name: Target collection name.
            rrf_k: The RRF constant parameter (default: 60).
        """
        self.store = store
        self.embedder = embedder
        self.collection_name = collection_name
        self.rrf_k = rrf_k

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[SearchResult],
        keyword_results: list[SearchResult],
        limit: int
    ) -> list[SearchResult]:
        """Combine two lists of search results using Reciprocal Rank Fusion."""
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        # Process dense results
        for rank, res in enumerate(dense_results):
            cid = res.chunk.id
            if cid:
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rank + 1 + self.rrf_k))
                chunk_map[cid] = res.chunk

        # Process keyword results
        for rank, res in enumerate(keyword_results):
            cid = res.chunk.id
            if cid:
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rank + 1 + self.rrf_k))
                chunk_map[cid] = res.chunk

        # Sort by score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        return [
            SearchResult(chunk=chunk_map[cid], score=rrf_scores[cid])
            for cid in sorted_ids[:limit]
        ]

    def _get_keyword_results(
        self,
        query: str,
        limit: int,
        filters: Filter | None
    ) -> list[SearchResult]:
        """Fetch keyword match results from Qdrant."""
        from qdrant_client.http import models as rest_models
        
        # Build a full-text match condition on the 'content' field
        keyword_condition = rest_models.FieldCondition(
            key="content",
            match=rest_models.MatchText(text=query)
        )
        
        # Parse existing filter if any
        base_filter = self.store._parse_filter(filters)
        
        # Merge conditions
        if base_filter:
            must_conditions = base_filter.must or []
            if not isinstance(must_conditions, list):
                must_conditions = [must_conditions]
            must_conditions.append(keyword_condition)
            qdrant_filter = rest_models.Filter(must=must_conditions)
        else:
            qdrant_filter = rest_models.Filter(must=[keyword_condition])

        try:
            # Query Qdrant with zero vector but using filter for keyword match.
            # To get hits ordered, we can query scroll or search.
            # Scroll is simpler for keyword filters, but search with a dummy vector
            # is also possible. Let's use Qdrant client scroll or count.
            # Actually, self.store.client.scroll is perfect here.
            hits, _ = self.store.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            results = []
            for hit in hits:
                payload = hit.payload or {}
                chunk = Chunk(
                    content=payload.get("content", ""),
                    document_id=payload.get("document_id"),
                    metadata=payload.get("metadata", {}),
                    id=str(hit.id)
                )
                # Flat score since it's boolean keyword match
                results.append(
                    SearchResult(chunk=chunk, score=1.0)
                )
            return results
        except Exception:
            return []

    async def _aget_keyword_results(
        self,
        query: str,
        limit: int,
        filters: Filter | None
    ) -> list[SearchResult]:
        """Fetch keyword match results asynchronously from Qdrant."""
        from qdrant_client.http import models as rest_models
        
        keyword_condition = rest_models.FieldCondition(
            key="content",
            match=rest_models.MatchText(text=query)
        )
        
        base_filter = self.store._parse_filter(filters)
        if base_filter:
            must_conditions = base_filter.must or []
            if not isinstance(must_conditions, list):
                must_conditions = [must_conditions]
            must_conditions.append(keyword_condition)
            qdrant_filter = rest_models.Filter(must=must_conditions)
        else:
            qdrant_filter = rest_models.Filter(must=[keyword_condition])

        try:
            hits, _ = await self.store.aclient.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            results = []
            for hit in hits:
                payload = hit.payload or {}
                chunk = Chunk(
                    content=payload.get("content", ""),
                    document_id=payload.get("document_id"),
                    metadata=payload.get("metadata", {}),
                    id=str(hit.id)
                )
                results.append(SearchResult(chunk=chunk, score=1.0))
            return results
        except Exception:
            return []

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        filters: Filter | None = None
    ) -> list[SearchResult]:
        """Retrieve relevant document chunks using hybrid search (Dense + Keyword)."""
        # Get dense search results
        query_vector = self.embedder.embed_query(query)
        dense_results = self.store.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            filters=filters
        )
        
        # Get keyword match results
        keyword_results = self._get_keyword_results(query, limit=limit, filters=filters)
        
        # Fuse results
        return self._reciprocal_rank_fusion(dense_results, keyword_results, limit)

    async def aretrieve(
        self,
        query: str,
        limit: int = 5,
        filters: Filter | None = None
    ) -> list[SearchResult]:
        """Retrieve relevant document chunks asynchronously using hybrid search."""
        # Run embedding and dense search
        query_vector = await self.embedder.aembed_query(query)
        dense_results = await self.store.asearch(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            filters=filters
        )
        
        # Run keyword search
        keyword_results = await self._aget_keyword_results(query, limit=limit, filters=filters)
        
        # Fuse results
        return self._reciprocal_rank_fusion(dense_results, keyword_results, limit)
