"""Local embedding generator wrapper using SentenceTransformers."""

import asyncio
from typing import Any

from rag_qdrant.embeddings.base import BaseEmbedder
from rag_qdrant.exceptions import EmbeddingError


class LocalEmbedder(BaseEmbedder):
    """Wrapper for local SentenceTransformer embeddings."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        **kwargs: Any
    ) -> None:
        """Initialize the LocalEmbedder.

        Args:
            model_name: The sentence-transformers model name to load.
            device: Device to use (e.g. "cpu", "cuda").
            model: Alternative name for model_name, for consistency with other embedders.
            api_key: Unused for local embedder, accepted for compatibility.
            **kwargs: Additional parameters passed to SentenceTransformer.
        """
        self.model_name = model or model_name
        self.device = device
        self.model_kwargs = kwargs
        self.api_key = api_key
        self._model: Any = None

    @property
    def model(self) -> Any:
        """Lazy load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise EmbeddingError(
                    "The 'sentence-transformers' package is required for LocalEmbedder. "
                    "Install it with: pip install 'rag-qdrant[local]'"
                ) from e
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    **self.model_kwargs
                )
            except Exception as e:
                raise EmbeddingError(
                    f"Failed to load sentence-transformers model {self.model_name}: {e}"
                ) from e
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents locally."""
        if not texts:
            return []
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return list(embeddings.tolist())
        except Exception as e:
            raise EmbeddingError(f"Local embedding generation failed: {e}") from e

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query locally."""
        try:
            embedding = self.model.encode(text, show_progress_bar=False)
            return list(embedding.tolist())
        except Exception as e:
            raise EmbeddingError(f"Local query embedding generation failed: {e}") from e

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents asynchronously using the event loop's executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        """Embed a single query asynchronously using the event loop's executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed_query, text)
