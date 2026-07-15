"""Fixed-size character chunker implementation."""

import asyncio
import uuid

from rag_qdrant.chunking.base import BaseChunker
from rag_qdrant.exceptions import ChunkingError
from rag_qdrant.interfaces.schemas import Chunk, Document


class FixedSizeChunker(BaseChunker):
    """Chunker that splits text into fixed character length chunks with overlap."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """Initialize the FixedSizeChunker.

        Args:
            chunk_size: The size of each chunk in characters.
            chunk_overlap: The overlap between consecutive chunks in characters.

        Raises:
            ChunkingError: If chunk_overlap is greater than or equal to chunk_size.
        """
        if chunk_overlap >= chunk_size:
            raise ChunkingError("chunk_overlap must be less than chunk_size")
        if chunk_size <= 0:
            raise ChunkingError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ChunkingError("chunk_overlap must be non-negative")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        """Split documents into fixed-size chunks.

        Args:
            documents: List of documents to split.

        Returns:
            List[Chunk]: The resulting list of chunks.
        """
        chunks: list[Chunk] = []
        for doc in documents:
            text = doc.content
            doc_id = doc.id or str(uuid.uuid4())
            
            if not text:
                continue

            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end]
                
                chunk_id = str(uuid.uuid4())
                
                # Copy document metadata and add chunking details
                chunk_metadata = doc.metadata.copy()
                chunk_metadata["chunk_start"] = start
                chunk_metadata["chunk_end"] = min(end, len(text))
                
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        metadata=chunk_metadata,
                        id=chunk_id,
                        document_id=doc_id
                    )
                )
                
                start += self.chunk_size - self.chunk_overlap

        return chunks

    async def achunk(self, documents: list[Document]) -> list[Chunk]:
        """Split documents asynchronously.

        Args:
            documents: List of documents to split.

        Returns:
            List[Chunk]: The resulting list of chunks.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.chunk, documents)
