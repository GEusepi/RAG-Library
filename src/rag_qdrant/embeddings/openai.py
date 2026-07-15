"""OpenAI embedding generator wrapper."""

from typing import Any

from rag_qdrant.embeddings.base import BaseEmbedder
from rag_qdrant.exceptions import EmbeddingError


class OpenAIEmbedder(BaseEmbedder):
    """Wrapper for OpenAI's embedding API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        **kwargs: Any
    ) -> None:
        """Initialize the OpenAIEmbedder.

        Args:
            api_key: OpenAI API key. If None, it will be read from the environment.
            model: OpenAI model to use.
            **kwargs: Additional parameters passed to the OpenAI client.
        """
        self.api_key = api_key
        self.model = model
        self.client_kwargs = kwargs
        self._client: Any = None
        self._aclient: Any = None

    @property
    def client(self) -> Any:
        """Lazy load the sync OpenAI client."""
        if self._client is None:
            try:
                import openai
            except ImportError as e:
                raise EmbeddingError(
                    "The 'openai' package is required for OpenAIEmbedder. "
                    "Install it with: pip install 'rag-qdrant[openai]'"
                ) from e
            self._client = openai.OpenAI(api_key=self.api_key, **self.client_kwargs)
        return self._client

    @property
    def aclient(self) -> Any:
        """Lazy load the async OpenAI client."""
        if self._aclient is None:
            try:
                import openai
            except ImportError as e:
                raise EmbeddingError(
                    "The 'openai' package is required for OpenAIEmbedder. "
                    "Install it with: pip install 'rag-qdrant[openai]'"
                ) from e
            self._aclient = openai.AsyncOpenAI(api_key=self.api_key, **self.client_kwargs)
        return self._aclient

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents using OpenAI."""
        if not texts:
            return []
        try:
            response = self.client.embeddings.create(input=texts, model=self.model)
            return [data.embedding for data in response.data]
        except Exception as e:
            raise EmbeddingError(f"OpenAI embedding generation failed: {e}") from e

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query using OpenAI."""
        try:
            response = self.client.embeddings.create(input=[text], model=self.model)
            return list(response.data[0].embedding)
        except Exception as e:
            raise EmbeddingError(f"OpenAI query embedding generation failed: {e}") from e

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents asynchronously using OpenAI."""
        if not texts:
            return []
        try:
            response = await self.aclient.embeddings.create(input=texts, model=self.model)
            return [data.embedding for data in response.data]
        except Exception as e:
            raise EmbeddingError(f"OpenAI async embedding generation failed: {e}") from e

    async def aembed_query(self, text: str) -> list[float]:
        """Embed a single query asynchronously using OpenAI."""
        try:
            response = await self.aclient.embeddings.create(input=[text], model=self.model)
            return list(response.data[0].embedding)
        except Exception as e:
            raise EmbeddingError(f"OpenAI async query embedding generation failed: {e}") from e
