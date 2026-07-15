"""Cohere embedding generator wrapper."""

from typing import Any

from rag_qdrant.embeddings.base import BaseEmbedder
from rag_qdrant.exceptions import EmbeddingError


class CohereEmbedder(BaseEmbedder):
    """Wrapper for Cohere's embedding API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "embed-english-v3.0",
        **kwargs: Any
    ) -> None:
        """Initialize the CohereEmbedder.

        Args:
            api_key: Cohere API key. If None, it will be read from the environment.
            model: Cohere model to use.
            **kwargs: Additional parameters passed to the Cohere client.
        """
        self.api_key = api_key
        self.model = model
        self.client_kwargs = kwargs
        self._client: Any = None
        self._aclient: Any = None

    @property
    def client(self) -> Any:
        """Lazy load the sync Cohere client."""
        if self._client is None:
            try:
                import cohere
            except ImportError as e:
                raise EmbeddingError(
                    "The 'cohere' package is required for CohereEmbedder. "
                    "Install it with: pip install 'rag-qdrant[cohere]'"
                ) from e
            self._client = cohere.Client(api_key=self.api_key, **self.client_kwargs)
        return self._client

    @property
    def aclient(self) -> Any:
        """Lazy load the async Cohere client."""
        if self._aclient is None:
            try:
                import cohere
            except ImportError as e:
                raise EmbeddingError(
                    "The 'cohere' package is required for CohereEmbedder. "
                    "Install it with: pip install 'rag-qdrant[cohere]'"
                ) from e
            self._aclient = cohere.AsyncClient(api_key=self.api_key, **self.client_kwargs)
        return self._aclient

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents using Cohere."""
        if not texts:
            return []
        try:
            response = self.client.embed(
                texts=texts,
                model=self.model,
                input_type="search_document",
                embedding_types=["float"]
            )
            # Response may return float embeddings or float_ list
            if hasattr(response.embeddings, "float_"):
                return list(response.embeddings.float_)
            return list(response.embeddings)
        except Exception as e:
            raise EmbeddingError(f"Cohere embedding generation failed: {e}") from e

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query using Cohere."""
        try:
            response = self.client.embed(
                texts=[text],
                model=self.model,
                input_type="search_query",
                embedding_types=["float"]
            )
            if hasattr(response.embeddings, "float_"):
                return list(response.embeddings.float_[0])
            return list(response.embeddings[0])
        except Exception as e:
            raise EmbeddingError(f"Cohere query embedding generation failed: {e}") from e

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents asynchronously using Cohere."""
        if not texts:
            return []
        try:
            response = await self.aclient.embed(
                texts=texts,
                model=self.model,
                input_type="search_document",
                embedding_types=["float"]
            )
            if hasattr(response.embeddings, "float_"):
                return list(response.embeddings.float_)
            return list(response.embeddings)
        except Exception as e:
            raise EmbeddingError(f"Cohere async embedding generation failed: {e}") from e

    async def aembed_query(self, text: str) -> list[float]:
        """Embed a single query asynchronously using Cohere."""
        try:
            response = await self.aclient.embed(
                texts=[text],
                model=self.model,
                input_type="search_query",
                embedding_types=["float"]
            )
            if hasattr(response.embeddings, "float_"):
                return list(response.embeddings.float_[0])
            return list(response.embeddings[0])
        except Exception as e:
            raise EmbeddingError(f"Cohere async query embedding generation failed: {e}") from e
