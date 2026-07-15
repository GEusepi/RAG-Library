"""Reranker implementation using Cohere API or local Cross-Encoders."""

import asyncio
from typing import Any

from rag_qdrant.exceptions import ConfigurationError
from rag_qdrant.interfaces.schemas import SearchResult


class Reranker:
    """Reranks search results using a cross-encoder model or Cohere API."""

    def __init__(
        self,
        provider: str = "local",
        model_name: str | None = None,
        api_key: str | None = None,
        **kwargs: Any
    ) -> None:
        """Initialize the Reranker.

        Args:
            provider: Reranking provider ('local' or 'cohere').
            model_name: Model name. Defaults to ms-marco-MiniLM for 'local'
                        and rerank-english-v3.0 for 'cohere'.
            api_key: Cohere API key if using 'cohere'.
            **kwargs: Extra parameters passed to the model initialization.
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.model_kwargs = kwargs
        
        if self.provider == "local":
            self.model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
            self._model: Any = None
        elif self.provider == "cohere":
            self.model_name = model_name or "rerank-english-v3.0"
            self._client: Any = None
            self._aclient: Any = None
        else:
            raise ConfigurationError(
                f"Unknown reranker provider: '{provider}'. "
                "Supported providers are: 'local', 'cohere'."
            )

    @property
    def model(self) -> Any:
        """Lazy load local CrossEncoder."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise ConfigurationError(
                    "The 'sentence-transformers' package is required for local Reranker. "
                    "Install it with: pip install 'rag-qdrant[local]'"
                ) from e
            self._model = CrossEncoder(self.model_name, **self.model_kwargs)
        return self._model

    @property
    def client(self) -> Any:
        """Lazy load Cohere client."""
        if self._client is None:
            try:
                import cohere
            except ImportError as e:
                raise ConfigurationError(
                    "The 'cohere' package is required for Cohere Reranker. "
                    "Install it with: pip install 'rag-qdrant[cohere]'"
                ) from e
            self._client = cohere.Client(api_key=self.api_key, **self.model_kwargs)
        return self._client

    @property
    def aclient(self) -> Any:
        """Lazy load async Cohere client."""
        if self._aclient is None:
            try:
                import cohere
            except ImportError as e:
                raise ConfigurationError(
                    "The 'cohere' package is required for Cohere Reranker. "
                    "Install it with: pip install 'rag-qdrant[cohere]'"
                ) from e
            self._aclient = cohere.AsyncClient(api_key=self.api_key, **self.model_kwargs)
        return self._aclient

    def _rerank_local(
        self, query: str, results: list[SearchResult], limit: int
    ) -> list[SearchResult]:
        if not results:
            return []
        
        pairs = [(query, res.chunk.content) for res in results]
        scores = self.model.predict(pairs)
        
        # Zip and sort by score descending
        ranked_results = [
            SearchResult(chunk=res.chunk, score=float(score))
            for res, score in zip(results, scores, strict=False)
        ]
        ranked_results.sort(key=lambda x: x.score, reverse=True)
        return ranked_results[:limit]

    def _rerank_cohere(
        self, query: str, results: list[SearchResult], limit: int
    ) -> list[SearchResult]:
        if not results:
            return []
            
        docs = [res.chunk.content for res in results]
        response = self.client.rerank(
            model=self.model_name,
            query=query,
            documents=docs,
            top_n=limit
        )
        
        ranked_results = []
        for result in response.results:
            orig = results[result.index]
            ranked_results.append(
                SearchResult(chunk=orig.chunk, score=float(result.relevance_score))
            )
        return ranked_results

    async def _arerank_cohere(
        self, query: str, results: list[SearchResult], limit: int
    ) -> list[SearchResult]:
        if not results:
            return []
            
        docs = [res.chunk.content for res in results]
        response = await self.aclient.rerank(
            model=self.model_name,
            query=query,
            documents=docs,
            top_n=limit
        )
        
        ranked_results = []
        for result in response.results:
            orig = results[result.index]
            ranked_results.append(
                SearchResult(chunk=orig.chunk, score=float(result.relevance_score))
            )
        return ranked_results

    def rerank(self, query: str, results: list[SearchResult], limit: int = 5) -> list[SearchResult]:
        """Rerank search results synchronously.

        Args:
            query: The user query text.
            results: List of SearchResult objects to rerank.
            limit: Maximum number of results to return.

        Returns:
            List[SearchResult]: Reranked search results.
        """
        if self.provider == "local":
            return self._rerank_local(query, results, limit)
        return self._rerank_cohere(query, results, limit)

    async def arerank(
        self, query: str, results: list[SearchResult], limit: int = 5
    ) -> list[SearchResult]:
        """Rerank search results asynchronously.

        Args:
            query: The user query text.
            results: List of SearchResult objects to rerank.
            limit: Maximum number of results to return.

        Returns:
            List[SearchResult]: Reranked search results.
        """
        if self.provider == "local":
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._rerank_local, query, results, limit)
        
        return await self._arerank_cohere(query, results, limit)
