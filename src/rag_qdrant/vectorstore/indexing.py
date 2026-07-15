"""Indexing utilities and helpers for vector stores."""


from rag_qdrant.interfaces.schemas import Chunk
from rag_qdrant.vectorstore.base import BaseVectorStore


def index_chunks(
    vector_store: BaseVectorStore,
    collection_name: str,
    chunks: list[Chunk],
    batch_size: int = 100
) -> None:
    """Index a list of chunks in batches.

    Args:
        vector_store: The vector store instance.
        collection_name: Target collection name.
        chunks: List of Chunk objects containing embeddings.
        batch_size: Size of each write batch.
    """
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vector_store.upsert(collection_name=collection_name, chunks=batch)


async def aindex_chunks(
    vector_store: BaseVectorStore,
    collection_name: str,
    chunks: list[Chunk],
    batch_size: int = 100
) -> None:
    """Index a list of chunks in batches asynchronously.

    Args:
        vector_store: The vector store instance.
        collection_name: Target collection name.
        chunks: List of Chunk objects containing embeddings.
        batch_size: Size of each write batch.
    """
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        await vector_store.aupsert(collection_name=collection_name, chunks=batch)
