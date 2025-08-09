# src/rag-mcp/document_processor.py
from pathlib import Path


class DocumentProcessor:
    """
    Handles extracting raw text from various file types.
    Initially supports plain text files.
    """

    def extract_text(self, file_path: Path) -> str:
        """
        Extracts text from a given file path.

        Args:
            file_path (Path): The path to the document.

        Returns:
            str: The extracted text content of the document.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file type is not supported.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            # Future: Add support for .pdf, .docx, etc.
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
