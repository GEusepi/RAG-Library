"""PDF loader implementation."""

import asyncio
from pathlib import Path

from rag_qdrant.exceptions import LoaderError
from rag_qdrant.interfaces.schemas import Document
from rag_qdrant.loaders.base import BaseLoader


class PDFLoader(BaseLoader):
    """Loader that reads documents from PDF files."""

    def __init__(self, file_path: str | Path) -> None:
        """Initialize the PDFLoader.

        Args:
            file_path: Path to the PDF file to read.
        """
        self.file_path = Path(file_path)

    def load(self) -> list[Document]:
        """Load the PDF file into a list of Documents, one per page.

        Returns:
            List[Document]: A list of Document objects representing pages.

        Raises:
            LoaderError: If the file does not exist or cannot be read.
        """
        try:
            import pypdf
        except ImportError as e:
            raise LoaderError(
                "The 'pypdf' package is required for PDFLoader. "
                "Install it with: pip install 'rag-qdrant[pdf]'"
            ) from e

        if not self.file_path.exists():
            raise LoaderError(f"File not found: {self.file_path}")

        try:
            documents: list[Document] = []
            with open(self.file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                total_pages = len(reader.pages)
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    documents.append(
                        Document(
                            content=text,
                            metadata={
                                "source": str(self.file_path),
                                "page": page_num + 1,
                                "total_pages": total_pages,
                            }
                        )
                    )
            return documents
        except Exception as e:
            raise LoaderError(f"Failed to read PDF file {self.file_path}: {e}") from e

    async def aload(self) -> list[Document]:
        """Load the PDF file asynchronously.

        Returns:
            List[Document]: A list of Document objects representing pages.

        Raises:
            LoaderError: If the file does not exist or cannot be read.
        """
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.load)
        except Exception as e:
            if isinstance(e, LoaderError):
                raise
            raise LoaderError(f"Async PDF load failed for file {self.file_path}: {e}") from e
