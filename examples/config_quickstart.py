"""Example script demonstrating the RAG pipeline loaded from a JSON configuration."""

import asyncio
import json
from pathlib import Path
from rag_qdrant import (
    TextLoader,
    RAGConfig,
    RAGPipeline
)


async def main():
    # 1. Create a dummy text file to index
    dummy_file = Path("sample_docs_config.txt")
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
    print("Created dummy document: sample_docs_config.txt")

    # Path to the JSON configuration file
    config_path = Path(__file__).parent / "pipeline_config.json"
    print(f"Loading pipeline configuration from: {config_path}")

    try:
        # 2. Load and parse the JSON configuration
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        # 3. Instantiate RAGConfig (Pydantic model) and RAGPipeline
        # We construct RAGConfig using model_validate (supported in Pydantic v2)
        config = RAGConfig.model_validate(config_data)
        
        print("Initializing the RAG pipeline from configuration...")
        pipeline = RAGPipeline.from_config(config)

        # 4. Setup loader
        loader = TextLoader(dummy_file)

        # Explicitly create the collection (this is optional as ingest auto-creates it,
        # but is demonstrated here to showcase the new collection management API)
        print(f"Creating collection '{pipeline.retriever.collection_name}'...")
        await pipeline.async_create_collection("test_collection")

        # 5. Ingest documents
        print("Ingesting documents into Qdrant...")
        await pipeline.async_ingest(loader)

        # 6. Query the database
        query_text = "What is Qdrant?"
        print(f"\nQuerying: '{query_text}'")
        
        # Get raw matching chunks
        results = await pipeline.async_retrieve(query_text)
        print("\nRetrieved matching chunks:")
        for i, res in enumerate(results):
            print(f"[Doc {i+1}] (Score: {res.score:.4f}): {res.chunk.content}")

        # Explicitly delete the collection after usage to clean up the database
        print(f"\nDeleting collection '{pipeline.retriever.collection_name}'...")
        await pipeline.async_delete_collection()

    finally:
        # Clean up the dummy file
        if dummy_file.exists():
            dummy_file.unlink()
            print("\nCleaned up dummy document.")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
