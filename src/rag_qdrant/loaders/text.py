"""Text loader implementation."""

import asyncio
from pathlib import Path

from rag_qdrant.exceptions import LoaderError
from rag_qdrant.interfaces.schemas import Document
from rag_qdrant.loaders.base import BaseLoader


class TextLoader(BaseLoader):
    """Loader that reads documents from plain text files."""

    def __init__(self, file_path: str | Path, encoding: str = "utf-8") -> None:
        """Initialize the TextLoader.

        Args:
            file_path: Path to the text file to read.
            encoding: Encoding to use when reading the file.
        """
        self.file_path = Path(file_path)
        self.encoding = encoding

    def load(self) -> list[Document]:
        """Load the text file into a Document.

        Returns:
            List[Document]: A list containing the loaded Document.

        Raises:
            LoaderError: If the file does not exist or cannot be read.
        """
        if not self.file_path.exists():
            raise LoaderError(f"File not found: {self.file_path}")
        
        try:
            content = self.file_path.read_text(encoding=self.encoding)
            return [
                Document(
                    content=content,
                    metadata={"source": str(self.file_path)}
                )
            ]
        except Exception as e:
            raise LoaderError(f"Failed to read file {self.file_path}: {e}") from e

    async def aload(self) -> list[Document]:
        """Load the text file asynchronously.

        Returns:
            List[Document]: A list containing the loaded Document.

        Raises:
            LoaderError: If the file does not exist or cannot be read.
        """
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.load)
        except Exception as e:
            if isinstance(e, LoaderError):
                raise
            raise LoaderError(f"Async load failed for file {self.file_path}: {e}") from e
