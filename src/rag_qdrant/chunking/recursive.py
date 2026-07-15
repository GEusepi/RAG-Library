"""Recursive character chunker implementation."""

import asyncio
import uuid

from rag_qdrant.chunking.base import BaseChunker
from rag_qdrant.exceptions import ChunkingError
from rag_qdrant.interfaces.schemas import Chunk, Document


class RecursiveCharacterChunker(BaseChunker):
    """Splits text recursively using a list of separators to keep related text together."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None
    ) -> None:
        """Initialize the RecursiveCharacterChunker.

        Args:
            chunk_size: Maximum size of a chunk in characters.
            chunk_overlap: Overlap between consecutive chunks.
            separators: List of separators to split on, ordered by priority.
                         Defaults to ["\n\n", "\n", " ", ""].

        Raises:
            ChunkingError: If parameters are invalid.
        """
        if chunk_overlap >= chunk_size:
            raise ChunkingError("chunk_overlap must be less than chunk_size")
        if chunk_size <= 0:
            raise ChunkingError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ChunkingError("chunk_overlap must be non-negative")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Split text recursively using separators."""
        final_chunks: list[str] = []
        
        # Get the first separator to try
        if not separators:
            # No separators left, force split by chunk size
            start = 0
            while start < len(text):
                final_chunks.append(text[start:start + self.chunk_size])
                start += self.chunk_size - self.chunk_overlap
            return final_chunks

        separator = separators[0]
        next_separators = separators[1:]
        
        # Split by separator
        splits = list(text) if separator == "" else text.split(separator)

        current_doc: list[str] = []
        current_len = 0

        for s in splits:
            # If the single split exceeds chunk_size, split it recursively
            if len(s) > self.chunk_size:
                # Flush the current doc first
                if current_doc:
                    final_chunks.append(separator.join(current_doc))
                    current_doc = []
                    current_len = 0
                
                # Recursively split the long segment
                sub_splits = self._split_text(s, next_separators)
                final_chunks.extend(sub_splits)
            elif current_len + len(s) + (len(separator) if current_doc else 0) <= self.chunk_size:
                current_doc.append(s)
                current_len += len(s) + (len(separator) if len(current_doc) > 1 else 0)
            else:
                # Flush current doc
                if current_doc:
                    final_chunks.append(separator.join(current_doc))
                
                # Start new doc with overlap if possible
                # Keep merging previous parts to construct overlap
                overlap_doc: list[str] = []
                overlap_len = 0
                for prev in reversed(current_doc):
                    sep_len = len(separator) if overlap_doc else 0
                    if overlap_len + len(prev) + sep_len <= self.chunk_overlap:
                        overlap_doc.insert(0, prev)
                        has_multiple = len(overlap_doc) > 1
                        overlap_len += len(prev) + (len(separator) if has_multiple else 0)
                    else:
                        break
                
                current_doc = overlap_doc + [s]
                sep_count = len(current_doc) - 1
                current_len = sum(len(x) for x in current_doc) + len(separator) * sep_count

        if current_doc:
            final_chunks.append(separator.join(current_doc))

        return final_chunks

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        """Split documents recursively.

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

            raw_chunks = self._split_text(text, self.separators)
            
            # Map raw string chunks to Chunk objects
            start_offset = 0
            for raw_chunk in raw_chunks:
                # Find start offset in text if possible to add to metadata
                offset = text.find(raw_chunk, start_offset)
                if offset != -1:
                    start_offset = offset
                
                chunk_id = str(uuid.uuid4())
                chunk_metadata = doc.metadata.copy()
                chunk_metadata["chunk_start"] = start_offset
                chunk_metadata["chunk_end"] = start_offset + len(raw_chunk)
                
                chunks.append(
                    Chunk(
                        content=raw_chunk,
                        metadata=chunk_metadata,
                        id=chunk_id,
                        document_id=doc_id
                    )
                )
                start_offset += len(raw_chunk) - self.chunk_overlap

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
