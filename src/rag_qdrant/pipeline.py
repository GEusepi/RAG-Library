"""Orchestrator for the RAG pipeline."""

import logging

from rag_qdrant.chunking.base import BaseChunker
from rag_qdrant.embeddings.base import BaseEmbedder
from rag_qdrant.exceptions import ConfigurationError
from rag_qdrant.interfaces.config import RAGConfig
from rag_qdrant.interfaces.schemas import SearchResult
from rag_qdrant.loaders.base import BaseLoader
from rag_qdrant.retrieval.hybrid import HybridRetriever
from rag_qdrant.retrieval.reranker import Reranker
from rag_qdrant.retrieval.retriever import Retriever
from rag_qdrant.vectorstore.base import BaseVectorStore
from rag_qdrant.vectorstore.indexing import aindex_chunks, index_chunks

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Orchestrates document ingestion and retrieval pipeline."""

    def __init__(
        self,
        store: BaseVectorStore,
        embedder: BaseEmbedder,
        chunker: BaseChunker,
        retriever: Retriever | HybridRetriever,
        reranker: Reranker | None = None
    ) -> None:
        """Initialize the RAGPipeline.

        Args:
            store: Vector store integration.
            embedder: Embedding generator.
            chunker: Document chunker.
            retriever: Retriever (dense or hybrid).
            reranker: Optional reranker.
        """
        self.store = store
        self.embedder = embedder
        self.chunker = chunker
        self.retriever = retriever
        self.reranker = reranker

    @classmethod
    def from_config(cls, config: RAGConfig) -> "RAGPipeline":
        """Instantiate a RAGPipeline from a RAGConfig object.

        Args:
            config: The complete RAG configurations.

        Returns:
            RAGPipeline: The configured pipeline.
        """
        # 1. Create Vector Store
        from rag_qdrant.vectorstore.factory import create_vector_store
        store = create_vector_store(
            provider=config.vectorstore.provider,
            location=config.vectorstore.location,
            url=config.vectorstore.url,
            host=config.vectorstore.host,
            port=config.vectorstore.port,
            api_key=config.vectorstore.api_key,
            **config.vectorstore.kwargs
        )

        # 2. Create Embedder
        from rag_qdrant.embeddings.factory import create_embedder
        embedder = create_embedder(
            provider=config.embeddings.provider,
            model=config.embeddings.model,
            api_key=config.embeddings.api_key,
            device=config.embeddings.device,
            **config.embeddings.kwargs
        )

        # 3. Create Chunker
        from rag_qdrant.chunking.factory import create_chunker
        chunker = create_chunker(
            chunker_type=config.chunker.type,
            chunk_size=config.chunker.chunk_size,
            chunk_overlap=config.chunker.chunk_overlap,
            separators=config.chunker.separators,
            threshold_percentile=config.chunker.threshold_percentile,
            min_chunk_size=config.chunker.min_chunk_size,
            embedder=embedder
        )

        # 4. Create Reranker (optional)
        reranker = None
        if config.retriever.reranker_provider:
            reranker = Reranker(
                provider=config.retriever.reranker_provider,
                model_name=config.retriever.reranker_model,
                api_key=config.retriever.reranker_api_key,
                **config.retriever.reranker_kwargs
            )

        # 5. Create Retriever
        retriever: Retriever | HybridRetriever
        if config.retriever.hybrid:
            # HybridRetriever requires QdrantVectorStore specifically
            from rag_qdrant.vectorstore.qdrant import QdrantVectorStore
            if not isinstance(store, QdrantVectorStore):
                raise ConfigurationError(
                    "HybridRetriever is currently only supported with QdrantVectorStore."
                )
            retriever = HybridRetriever(
                store=store,
                embedder=embedder,
                collection_name=config.retriever.collection_name,
                rrf_k=config.retriever.rrf_k
            )
        else:
            retriever = Retriever(
                store=store,
                embedder=embedder,
                collection_name=config.retriever.collection_name
            )

        return cls(
            store=store,
            embedder=embedder,
            chunker=chunker,
            retriever=retriever,
            reranker=reranker
        )

    def create_collection(
        self,
        collection_name: str | None = None,
        vector_size: int | None = None,
        distance: str = "Cosine"
    ) -> None:
        """Create a vector collection.

        Args:
            collection_name: Optional custom collection name.
                             Defaults to retriever's collection name.
            vector_size: Optional vector size. Defaults to embedder's output size.
            distance: Similarity distance metric.
        """
        col_name = collection_name or self.retriever.collection_name
        if vector_size is None:
            vector_size = len(self.embedder.embed_query("dummy"))
        self.store.create_collection(
            collection_name=col_name,
            vector_size=vector_size,
            distance=distance
        )

    async def async_create_collection(
        self,
        collection_name: str | None = None,
        vector_size: int | None = None,
        distance: str = "Cosine"
    ) -> None:
        """Create a vector collection asynchronously.

        Args:
            collection_name: Optional custom collection name.
                             Defaults to retriever's collection name.
            vector_size: Optional vector size. Defaults to embedder's output size.
            distance: Similarity distance metric.
        """
        col_name = collection_name or self.retriever.collection_name
        if vector_size is None:
            emb = await self.embedder.aembed_query("dummy")
            vector_size = len(emb)
        await self.store.acreate_collection(
            collection_name=col_name,
            vector_size=vector_size,
            distance=distance
        )

    def delete_collection(self, collection_name: str | None = None) -> None:
        """Delete a vector collection.

        Args:
            collection_name: Optional custom collection name.
                             Defaults to retriever's collection name.
        """
        col_name = collection_name or self.retriever.collection_name
        self.store.delete_collection(collection_name=col_name)

    async def async_delete_collection(self, collection_name: str | None = None) -> None:
        """Delete a vector collection asynchronously.

        Args:
            collection_name: Optional custom collection name.
                             Defaults to retriever's collection name.
        """
        col_name = collection_name or self.retriever.collection_name
        await self.store.adelete_collection(collection_name=col_name)

    def ingest(self, loader: BaseLoader, batch_size: int = 100) -> None:
        """Load, chunk, embed, and index documents.

        Usage:
            >>> pipeline.ingest(loader=TextLoader("data.txt"), batch_size=50)

        If the target collection does not exist, it will be automatically created.

        Args:
            loader: The document loader to retrieve source docs.
            batch_size: Batch size for indexing database writes.
        """
        logger.info("Starting ingestion workflow...")
        documents = loader.load()
        if not documents:
            logger.warning("No documents loaded.")
            return

        chunks = self.chunker.chunk(documents)
        if not chunks:
            logger.warning("No chunks created from documents.")
            return

        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.embed_documents(texts)
        for chunk, emb in zip(chunks, embeddings, strict=False):
            chunk.embedding = emb

        # Resolve collection name
        collection_name = self.retriever.collection_name
        vector_size = len(embeddings[0])

        # Auto create collection if not exists (checked via try/except on creation)
        try:
            self.store.create_collection(
                collection_name=collection_name,
                vector_size=vector_size
            )
            logger.info(f"Created new collection '{collection_name}' with size {vector_size}.")
        except Exception:
            # Collection already exists or failed to create
            pass

        # Index in batches
        index_chunks(
            vector_store=self.store,
            collection_name=collection_name,
            chunks=chunks,
            batch_size=batch_size
        )
        logger.info(f"Ingested {len(chunks)} chunks into '{collection_name}'.")

    async def async_ingest(self, loader: BaseLoader, batch_size: int = 100) -> None:
        """Load, chunk, embed, and index documents asynchronously.

        Usage:
            >>> await pipeline.aingest(loader=TextLoader("data.txt"), batch_size=50)

        If the target collection does not exist, it will be automatically created.

        Args:
            loader: The document loader to retrieve source docs.
            batch_size: Batch size for indexing database writes.
        """
        logger.info("Starting async ingestion workflow...")
        documents = await loader.aload()
        if not documents:
            logger.warning("No documents loaded.")
            return

        chunks = await self.chunker.achunk(documents)
        if not chunks:
            logger.warning("No chunks created from documents.")
            return

        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = await self.embedder.aembed_documents(texts)
        for chunk, emb in zip(chunks, embeddings, strict=False):
            chunk.embedding = emb

        collection_name = self.retriever.collection_name
        vector_size = len(embeddings[0])

        try:
            await self.store.acreate_collection(
                collection_name=collection_name,
                vector_size=vector_size
            )
            logger.info(f"Created new collection '{collection_name}' asynchronously.")
        except Exception:
            pass

        await aindex_chunks(
            vector_store=self.store,
            collection_name=collection_name,
            chunks=chunks,
            batch_size=batch_size
        )
        logger.info(f"Ingested {len(chunks)} chunks into '{collection_name}' asynchronously.")

    def retrieve(self, question: str) -> list[SearchResult]:
        """Retrieve relevant context and optionally rerank it.

        Usage:
            >>> results = pipeline.query("What is Qdrant?")
            >>> for res in results:
            >>>     print(f"Content: {res.chunk.content}, Score: {res.score}")

        Args:
            question: The user query text.

        Returns:
            List[SearchResult]: Relevant search results.
        """
        # 1. Retrieve
        results = self.retriever.retrieve(
            query=question,
            limit=self.retriever.limit if hasattr(self.retriever, "limit") else 5
        )

        # 2. Rerank
        if self.reranker and results:
            limit_val = self.retriever.limit if hasattr(self.retriever, "limit") else 5
            results = self.reranker.rerank(query=question, results=results, limit=limit_val)

        return results

    async def async_retrieve(self, question: str) -> list[SearchResult]:
        """Retrieve relevant context and optionally rerank it asynchronously.

        Usage:
            >>> results = await pipeline.aquery("What is Qdrant?")
            >>> for res in results:
            >>>     print(f"Content: {res.chunk.content}, Score: {res.score}")

        Args:
            question: The user query text.

        Returns:
            List[SearchResult]: Relevant search results.
        """
        # 1. Retrieve
        results = await self.retriever.aretrieve(
            query=question,
            limit=self.retriever.limit if hasattr(self.retriever, "limit") else 5
        )

        # 2. Rerank
        if self.reranker and results:
            limit_val = self.retriever.limit if hasattr(self.retriever, "limit") else 5
            results = await self.reranker.arerank(query=question, results=results, limit=limit_val)

        return results
