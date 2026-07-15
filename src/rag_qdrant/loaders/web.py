"""Web page loader implementation."""


from rag_qdrant.exceptions import LoaderError
from rag_qdrant.interfaces.schemas import Document
from rag_qdrant.loaders.base import BaseLoader


class WebLoader(BaseLoader):
    """Loader that fetches and extracts text from web URLs."""

    def __init__(self, url: str, timeout_seconds: int = 10) -> None:
        """Initialize the WebLoader.

        Args:
            url: The URL to fetch content from.
            timeout_seconds: HTTP request timeout in seconds.
        """
        self.url = url
        self.timeout_seconds = timeout_seconds

    def load(self) -> list[Document]:
        """Fetch and extract text from the URL synchronously.

        Returns:
            List[Document]: A list containing the loaded Document.

        Raises:
            LoaderError: If HTTP request fails or parsing fails.
        """
        try:
            import httpx
            from bs4 import BeautifulSoup
        except ImportError as e:
            raise LoaderError(
                "The 'httpx' and 'beautifulsoup4' packages are required for WebLoader. "
                "Install them with: pip install 'rag-qdrant[web]'"
            ) from e

        try:
            response = httpx.get(self.url, timeout=self.timeout_seconds, follow_redirects=True)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            text = soup.get_text(separator="\n")
            # Clean up whitespace
            cleaned_text = "\n".join(
                [line.strip() for line in text.splitlines() if line.strip()]
            )
            
            title = soup.title.string.strip() if soup.title else ""
            
            return [
                Document(
                    content=cleaned_text,
                    metadata={"source": self.url, "title": title}
                )
            ]
        except Exception as e:
            raise LoaderError(f"Failed to fetch or parse web content from {self.url}: {e}") from e

    async def aload(self) -> list[Document]:
        """Fetch and extract text from the URL asynchronously.

        Returns:
            List[Document]: A list containing the loaded Document.

        Raises:
            LoaderError: If HTTP request fails or parsing fails.
        """
        try:
            import httpx
            from bs4 import BeautifulSoup
        except ImportError as e:
            raise LoaderError(
                "The 'httpx' and 'beautifulsoup4' packages are required for WebLoader. "
                "Install them with: pip install 'rag-qdrant[web]'"
            ) from e

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds, follow_redirects=True
            ) as client:
                response = await client.get(self.url)
                response.raise_for_status()
                
            # Run parser in executor to avoid blocking the event loop
            import asyncio
            loop = asyncio.get_running_loop()
            
            def parse() -> list[Document]:
                soup = BeautifulSoup(response.content, "html.parser")
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text(separator="\n")
                cleaned_text = "\n".join(
                    [line.strip() for line in text.splitlines() if line.strip()]
                )
                title = soup.title.string.strip() if soup.title else ""
                return [
                    Document(
                        content=cleaned_text,
                        metadata={"source": self.url, "title": title}
                    )
                ]
                
            return await loop.run_in_executor(None, parse)
        except Exception as e:
            raise LoaderError(f"Async fetch or parse failed for {self.url}: {e}") from e
