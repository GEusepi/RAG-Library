"""Factory for creating text chunkers."""

from typing import Any

from rag_qdrant.chunking.base import BaseChunker
from rag_qdrant.exceptions import ConfigurationError


def create_chunker(chunker_type: str, **kwargs: Any) -> BaseChunker:
    """Create a chunker instance based on the chunker type.

    Usage:
        >>> # Fixed size chunker
        >>> chunker = create_chunker("fixed", chunk_size=500, chunk_overlap=50)
        
        >>> # Recursive Character Chunker
        >>> chunker = create_chunker(
        ...     "recursive", chunk_size=1000, chunk_overlap=100, separators=["\n\n"]
        ... )

        >>> # Semantic Chunker (requires an embedder instance)
        >>> chunker = create_chunker("semantic", embedder=embedder, threshold_percentile=95.0)

    Args:
        chunker_type: The chunker type ('fixed', 'recursive', 'semantic').
        **kwargs: Configuration arguments passed to the chunker constructor.

    Returns:
        BaseChunker: An instance of the requested chunker.

    Raises:
        ConfigurationError: If the chunker type is unknown or invalid.
    """
    chunker_type_lower = chunker_type.lower()

    if chunker_type_lower == "fixed":
        from rag_qdrant.chunking.fixed_size import FixedSizeChunker
        # Extract only kwargs accepted by FixedSizeChunker
        fixed_kwargs = {
            k: v for k, v in kwargs.items()
            if k in {"chunk_size", "chunk_overlap"} and v is not None
        }
        return FixedSizeChunker(**fixed_kwargs)

    if chunker_type_lower == "recursive":
        from rag_qdrant.chunking.recursive import RecursiveCharacterChunker
        # Extract only kwargs accepted by RecursiveCharacterChunker
        rec_kwargs = {
            k: v for k, v in kwargs.items()
            if k in {"chunk_size", "chunk_overlap", "separators"} and v is not None
        }
        return RecursiveCharacterChunker(**rec_kwargs)

    if chunker_type_lower == "semantic":
        from rag_qdrant.chunking.semantic import SemanticChunker
        # Extract only kwargs accepted by SemanticChunker
        sem_kwargs = {
            k: v for k, v in kwargs.items()
            if k in {"embedder", "threshold_percentile", "min_chunk_size"} and v is not None
        }
        if "embedder" not in sem_kwargs:
            raise ConfigurationError("The 'embedder' parameter is required for SemanticChunker.")
        return SemanticChunker(**sem_kwargs)

    raise ConfigurationError(
        f"Unknown chunker type: '{chunker_type}'. "
        "Supported chunkers are: 'fixed', 'recursive', 'semantic'."
    )
