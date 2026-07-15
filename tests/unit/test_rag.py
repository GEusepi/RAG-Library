"""Unit tests for the RAG library components."""

import tempfile
from pathlib import Path
import pytest

from rag_qdrant.loaders.text import TextLoader
from rag_qdrant.chunking.fixed_size import FixedSizeChunker
from rag_qdrant.chunking.recursive import RecursiveCharacterChunker
from rag_qdrant.retrieval.retriever import Retriever
from rag_qdrant.pipeline import RAGPipeline
from rag_qdrant.exceptions import LoaderError, ChunkingError


def test_text_loader_sync():
    """Verify TextLoader synchronously reads content."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Hello World!")
        temp_path = f.name

    try:
        loader = TextLoader(temp_path)
        docs = loader.load()
        assert len(docs) == 1
        assert docs[0].content == "Hello World!"
        assert docs[0].metadata["source"] == temp_path
    finally:
        Path(temp_path).unlink()


@pytest.mark.asyncio
async def test_text_loader_async():
    """Verify TextLoader asynchronously reads content."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Hello Async World!")
        temp_path = f.name
 
    try:
        loader = TextLoader(temp_path)
        docs = await loader.aload()
        assert len(docs) == 1
        assert docs[0].content == "Hello Async World!"
    finally:
        Path(temp_path).unlink()


def test_text_loader_not_found():
    """Verify TextLoader raises LoaderError if file is missing."""
    loader = TextLoader("non_existent_file_xyz.txt")
    with pytest.raises(LoaderError):
        loader.load()


def test_fixed_size_chunker():
    """Verify FixedSizeChunker splits text at character boundaries."""
    chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=2)
    from rag_qdrant.interfaces.schemas import Document
    doc = Document(content="abcdefghijklmnop", metadata={"source": "test"})
    chunks = chunker.chunk([doc])
    
    assert len(chunks) > 1
    # Check overlap and chunks configuration
    assert chunks[0].content == "abcdefghij"
    assert chunks[1].content == "ijklmnop"  # 10 - 2 overlap = offset 8, so ij + klmnop
    assert chunks[0].document_id is not None


def test_recursive_character_chunker():
    """Verify RecursiveCharacterChunker splits by separators."""
    chunker = RecursiveCharacterChunker(chunk_size=15, chunk_overlap=2, separators=["\n\n", "\n", " "])
    from rag_qdrant.interfaces.schemas import Document
    doc = Document(content="hello world\n\nhow are you doing today", metadata={"source": "test"})
    chunks = chunker.chunk([doc])
    
    assert len(chunks) >= 2
    # Ensure separator priority is respected
    assert "hello world" in chunks[0].content


def test_retriever(fake_vector_store, fake_embedder):
    """Verify Retriever correctly queries the vector store."""
    from rag_qdrant.interfaces.schemas import Chunk
    
    # Pre-populate fake store
    chunk = Chunk(content="test content", id="123", embedding=[0.1, 0.2, 0.3])
    fake_vector_store.create_collection("test_col", 3)
    fake_vector_store.upsert("test_col", [chunk])
    
    retriever = Retriever(store=fake_vector_store, embedder=fake_embedder, collection_name="test_col")
    results = retriever.retrieve("query")
    
    assert len(results) == 1
    assert results[0].chunk.content == "test content"
    assert results[0].score == 1.0


@pytest.mark.asyncio
async def test_pipeline_ingest_and_query(fake_vector_store, fake_embedder, fake_loader, fake_chunker):
    """Verify RAGPipeline coordinates full ingestion and querying workflow."""
    retriever = Retriever(store=fake_vector_store, embedder=fake_embedder, collection_name="my_col")
    
    pipeline = RAGPipeline(
        store=fake_vector_store,
        embedder=fake_embedder,
        chunker=fake_chunker,
        retriever=retriever
    )
    
    # Ingest using fake loader
    await pipeline.aingest(fake_loader)
    
    # Query using pipeline
    results = await pipeline.aquery("test question")
    
    # Check that we retrieved the fake document content
    assert len(results) == 1
    assert results[0].chunk.content == "Fake content"


def test_chunker_factory(fake_embedder):
    """Verify that create_chunker instantiates chunkers correctly."""
    from rag_qdrant.chunking.factory import create_chunker
    from rag_qdrant.chunking.fixed_size import FixedSizeChunker
    from rag_qdrant.chunking.recursive import RecursiveCharacterChunker
    from rag_qdrant.chunking.semantic import SemanticChunker
    from rag_qdrant.exceptions import ConfigurationError

    fixed = create_chunker("fixed", chunk_size=100, chunk_overlap=10)
    assert isinstance(fixed, FixedSizeChunker)

    recursive = create_chunker("recursive", chunk_size=100, chunk_overlap=10, separators=["\n"])
    assert isinstance(recursive, RecursiveCharacterChunker)

    semantic = create_chunker("semantic", embedder=fake_embedder, threshold_percentile=90.0, min_chunk_size=50)
    assert isinstance(semantic, SemanticChunker)

    with pytest.raises(ConfigurationError):
        create_chunker("unknown_type")
