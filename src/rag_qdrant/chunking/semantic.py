"""Semantic chunker implementation based on sentence embedding similarity."""

import asyncio
import re
import uuid

import numpy as np

from rag_qdrant.chunking.base import BaseChunker
from rag_qdrant.embeddings.base import BaseEmbedder
from rag_qdrant.exceptions import ChunkingError
from rag_qdrant.interfaces.schemas import Chunk, Document


class SemanticChunker(BaseChunker):
    """Splits documents into chunks at points of semantic shifts based on embedding similarity."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        threshold_percentile: float = 95.0,
        min_chunk_size: int = 100,
    ) -> None:
        """Initialize the SemanticChunker.

        Args:
            embedder: The embedder used to embed sentences.
            threshold_percentile: The percentile of differences above which a split is created.
            min_chunk_size: Minimum characters required to consider splitting
                            (to avoid too small chunks).
        """
        self.embedder = embedder
        self.threshold_percentile = threshold_percentile
        self.min_chunk_size = min_chunk_size

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences using simple regex patterns."""
        # Split by periods, exclamation marks, or question marks followed by space or newline
        sentence_end = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_end.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _calculate_cosine_distances(self, embeddings: list[list[float]]) -> list[float]:
        """Calculate cosine distances between consecutive embeddings."""
        distances = []
        for i in range(len(embeddings) - 1):
            emb1 = np.array(embeddings[i])
            emb2 = np.array(embeddings[i + 1])
            
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            similarity = 0.0 if norm1 == 0 or norm2 == 0 else np.dot(emb1, emb2) / (norm1 * norm2)
                
            distances.append(1.0 - similarity)
        return distances

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        """Split documents into semantically coherent chunks.

        Args:
            documents: List of documents to split.

        Returns:
            List[Chunk]: The resulting list of chunks.

        Raises:
            ChunkingError: If embedding generation or split calculation fails.
        """
        chunks: list[Chunk] = []
        for doc in documents:
            text = doc.content
            doc_id = doc.id or str(uuid.uuid4())
            
            if not text or len(text) < self.min_chunk_size:
                # Document is too short, keep it as one chunk
                chunk_id = str(uuid.uuid4())
                chunks.append(
                    Chunk(
                        content=text,
                        metadata=doc.metadata.copy(),
                        id=chunk_id,
                        document_id=doc_id
                    )
                )
                continue

            sentences = self._split_into_sentences(text)
            if len(sentences) <= 1:
                chunk_id = str(uuid.uuid4())
                chunks.append(
                    Chunk(
                        content=text,
                        metadata=doc.metadata.copy(),
                        id=chunk_id,
                        document_id=doc_id
                    )
                )
                continue

            try:
                # Generate embeddings for sentences
                embeddings = self.embedder.embed_documents(sentences)
            except Exception as e:
                raise ChunkingError(
                    f"Embedding generation failed for semantic chunking: {e}"
                ) from e

            # Calculate differences/distances
            distances = self._calculate_cosine_distances(embeddings)
            
            # Determine threshold
            threshold = np.percentile(distances, self.threshold_percentile)
            
            # Split sentences at indices where distance > threshold
            split_indices = [i for i, dist in enumerate(distances) if dist > threshold]
            
            # Reconstruct chunks
            current_chunk_sentences = []
            current_length = 0
            
            for idx, sentence in enumerate(sentences):
                current_chunk_sentences.append(sentence)
                current_length += len(sentence)
                
                # If we hit a split index and the current chunk is sufficiently large, split
                if idx in split_indices and current_length >= self.min_chunk_size:
                    chunk_text = " ".join(current_chunk_sentences)
                    chunk_id = str(uuid.uuid4())
                    
                    chunks.append(
                        Chunk(
                            content=chunk_text,
                            metadata=doc.metadata.copy(),
                            id=chunk_id,
                            document_id=doc_id
                        )
                    )
                    current_chunk_sentences = []
                    current_length = 0
            
            # Add remaining sentences
            if current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                chunk_id = str(uuid.uuid4())
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        metadata=doc.metadata.copy(),
                        id=chunk_id,
                        document_id=doc_id
                    )
                )

        return chunks

    async def achunk(self, documents: list[Document]) -> list[Chunk]:
        """Split documents into semantically coherent chunks asynchronously.

        Args:
            documents: List of documents to split.

        Returns:
            List[Chunk]: The resulting list of chunks.

        Raises:
            ChunkingError: If embedding generation or split calculation fails.
        """
        # We can implement proper async by using the async variant of the embedder!
        chunks: list[Chunk] = []
        for doc in documents:
            text = doc.content
            doc_id = doc.id or str(uuid.uuid4())
            
            if not text or len(text) < self.min_chunk_size:
                chunk_id = str(uuid.uuid4())
                chunks.append(
                    Chunk(
                        content=text,
                        metadata=doc.metadata.copy(),
                        id=chunk_id,
                        document_id=doc_id
                    )
                )
                continue

            sentences = self._split_into_sentences(text)
            if len(sentences) <= 1:
                chunk_id = str(uuid.uuid4())
                chunks.append(
                    Chunk(
                        content=text,
                        metadata=doc.metadata.copy(),
                        id=chunk_id,
                        document_id=doc_id
                    )
                )
                continue

            try:
                # Use async embed documents
                embeddings = await self.embedder.aembed_documents(sentences)
            except Exception as e:
                raise ChunkingError(
                    f"Async embedding generation failed for semantic chunking: {e}"
                ) from e

            # Run mathematical/CPU-bound calculations in executor
            loop = asyncio.get_running_loop()
            
            def calculate_splits(
                embeddings: list[list[float]] = embeddings,
                sentences: list[str] = sentences,
                doc: Document = doc,
                doc_id: str = doc_id
            ) -> list[Chunk]:
                distances = self._calculate_cosine_distances(embeddings)
                threshold = np.percentile(distances, self.threshold_percentile)
                split_indices = [i for i, dist in enumerate(distances) if dist > threshold]
                
                doc_chunks = []
                current_chunk_sentences = []
                current_length = 0
                
                for idx, sentence in enumerate(sentences):
                    current_chunk_sentences.append(sentence)
                    current_length += len(sentence)
                    
                    if idx in split_indices and current_length >= self.min_chunk_size:
                        chunk_text = " ".join(current_chunk_sentences)
                        chunk_id = str(uuid.uuid4())
                        doc_chunks.append(
                            Chunk(
                                content=chunk_text,
                                metadata=doc.metadata.copy(),
                                id=chunk_id,
                                document_id=doc_id
                            )
                        )
                        current_chunk_sentences = []
                        current_length = 0
                
                if current_chunk_sentences:
                    chunk_text = " ".join(current_chunk_sentences)
                    chunk_id = str(uuid.uuid4())
                    doc_chunks.append(
                        Chunk(
                            content=chunk_text,
                            metadata=doc.metadata.copy(),
                            id=chunk_id,
                            document_id=doc_id
                        )
                    )
                return doc_chunks
                
            doc_chunks = await loop.run_in_executor(None, calculate_splits)
            chunks.extend(doc_chunks)

        return chunks
