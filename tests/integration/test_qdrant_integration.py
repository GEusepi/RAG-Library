"""Integration tests for the QdrantVectorStore using in-memory QdrantClient."""

import pytest
from rag_qdrant.vectorstore.qdrant import QdrantVectorStore
from rag_qdrant.interfaces.schemas import Chunk, Filter
from rag_qdrant.exceptions import CollectionNotFoundError, VectorStoreError


def test_qdrant_sync_lifecycle():
    """Verify sync collection creation, upsert, search, and delete."""
    store = QdrantVectorStore(location=":memory:")
    collection_name = "test_sync_collection"
    
    # 1. Create collection
    store.create_collection(collection_name=collection_name, vector_size=3, distance="Cosine")
    
    # 2. Upsert chunks
    chunk1 = Chunk(content="semantic search engine", id="1", embedding=[1.0, 0.0, 0.0], metadata={"category": "tech"})
    chunk2 = Chunk(content="cooking recipe", id="2", embedding=[0.0, 1.0, 0.0], metadata={"category": "cooking"})
    store.upsert(collection_name=collection_name, chunks=[chunk1, chunk2])
    
    # 3. Search without filter
    results = store.search(collection_name=collection_name, query_vector=[1.0, 0.1, 0.0], limit=1)
    assert len(results) == 1
    assert results[0].chunk.content == "semantic search engine"
    assert results[0].chunk.id == "1"
    
    # 4. Search with filter
    filter_cooking = Filter(metadata_filter={"category": "cooking"})
    results_cooking = store.search(
        collection_name=collection_name,
        query_vector=[0.1, 1.0, 0.0],
        limit=1,
        filters=filter_cooking
    )
    assert len(results_cooking) == 1
    assert results_cooking[0].chunk.content == "cooking recipe"
    
    # 5. Delete with filter
    store.delete(collection_name=collection_name, filters=filter_cooking)
    
    # Verify deletion
    results_after_delete = store.search(
        collection_name=collection_name,
        query_vector=[0.0, 1.0, 0.0],
        limit=10,
        filters=filter_cooking
    )
    assert len(results_after_delete) == 0


@pytest.mark.asyncio
async def test_qdrant_async_lifecycle():
    """Verify async collection creation, upsert, search, and delete."""
    store = QdrantVectorStore(location=":memory:")
    collection_name = "test_async_collection"
    
    # 1. Create collection
    await store.acreate_collection(collection_name=collection_name, vector_size=3, distance="Cosine")
    
    # 2. Upsert chunks
    chunk1 = Chunk(content="async programming in python", id="1", embedding=[1.0, 0.0, 0.0], metadata={"lang": "python"})
    chunk2 = Chunk(content="javascript promises", id="2", embedding=[0.0, 1.0, 0.0], metadata={"lang": "js"})
    await store.aupsert(collection_name=collection_name, chunks=[chunk1, chunk2])
    
    # 3. Search
    results = await store.asearch(collection_name=collection_name, query_vector=[0.9, 0.1, 0.0], limit=1)
    assert len(results) == 1
    assert results[0].chunk.content == "async programming in python"
    
    # 4. Delete
    filter_js = Filter(metadata_filter={"lang": "js"})
    await store.adelete(collection_name=collection_name, filters=filter_js)
    
    # Verify deletion
    results_after_delete = await store.asearch(
        collection_name=collection_name,
        query_vector=[0.0, 1.0, 0.0],
        limit=10,
        filters=filter_js
    )
    assert len(results_after_delete) == 0


def test_qdrant_missing_collection():
    """Verify appropriate exception is raised when collection doesn't exist."""
    store = QdrantVectorStore(location=":memory:")
    with pytest.raises(CollectionNotFoundError):
        store.search("missing_collection_xyz", [1.0, 0.0, 0.0])
