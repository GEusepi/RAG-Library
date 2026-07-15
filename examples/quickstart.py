"""Quickstart example script demonstrating the RAG pipeline."""

import asyncio
from pathlib import Path
from rag_qdrant import (
    TextLoader,
    RecursiveCharacterChunker,
    create_embedder,
    create_vector_store,
    Retriever,
    RAGPipeline
)


async def main():
    # 1. Create a dummy text file to index
    dummy_file = Path("sample_docs.txt")
    dummy_file.write_text(
        "RAG (Retrieval-Augmented Generation) is an AI framework for retrieving facts "
        "from an external knowledge base to ground large language models (LLMs) on standard, "
        "factual, and up-to-date information.\n\n"
        "Qdrant is a vector similarity search engine. It provides a production-ready "
        "service with a convenient API to store, search, and manage points (vectors with payload).\n\n"
        "This library combines Qdrant with modular ingestion and LLM generation interfaces "
        "to make building production-grade RAG applications straightforward.",
        encoding="utf-8"
    )
    print("Created dummy document: sample_docs.txt")

    try:
        # 2. Setup loader and chunker
        loader = TextLoader(dummy_file)
        chunker = RecursiveCharacterChunker(chunk_size=150, chunk_overlap=20)

        # 3. Setup local embeddings and in-memory Qdrant store
        # We use 'local' embedding provider which runs sentence-transformers.
        # We use ':memory:' for Qdrant to run it without any local docker container requirement!
        print("Initializing local embedder (SentenceTransformers) and in-memory Qdrant...")
        embedder = create_embedder("local", model_name="all-MiniLM-L6-v2")
        store = create_vector_store("qdrant", location=":memory:")

        # 4. Setup Retriever
        collection_name = "demo_collection"
        retriever = Retriever(store=store, embedder=embedder, collection_name=collection_name)

        # 5. Initialize the RAGPipeline (leaving generator to None to return raw retrieved documents)
        pipeline = RAGPipeline(
            store=store,
            embedder=embedder,
            chunker=chunker,
            retriever=retriever
        )

        # 6. Ingest documents
        print("Ingesting documents into Qdrant...")
        await pipeline.aingest(loader)

        # 7. Query the database
        query_text = "What is Qdrant?"
        print(f"\nQuerying: '{query_text}'")
        
        # Get raw matching chunks
        results = await pipeline.aquery(query_text)
        print("\nRetrieved matching chunks:")
        for i, res in enumerate(results):
            print(f"[Doc {i+1}] (Score: {res.score:.4f}): {res.chunk.content}")

    finally:
        # Clean up the dummy file
        if dummy_file.exists():
            dummy_file.unlink()
            print("\nCleaned up dummy document.")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
