# tests/test_document_processor.py
import pytest
from pathlib import Path
from src.rag_mcp.document_processor import DocumentProcessor

@pytest.fixture
def doc_processor():
    """Provides a DocumentProcessor instance for tests."""
    return DocumentProcessor()

@pytest.fixture
def temp_txt_file(tmp_path):
    """Creates a temporary .txt file for testing."""
    content = "This is a test document.\nIt has multiple lines."
    file_path = tmp_path / "test_doc.txt"
    file_path.write_text(content)
    return file_path, content

@pytest.fixture
def temp_unsupported_file(tmp_path):
    """Creates a temporary file with an unsupported extension."""
    file_path = tmp_path / "test_doc.pdf" # Using .pdf as an example of unsupported
    file_path.write_text("PDF content simulation")
    return file_path

def test_extract_text_txt(doc_processor, temp_txt_file):
    """Tests text extraction from a plain .txt file."""
    file_path, expected_content = temp_txt_file
    extracted_text = doc_processor.extract_text(file_path)
    assert extracted_text == expected_content

def test_extract_text_file_not_found(doc_processor, tmp_path):
    """Tests handling of a non-existent file."""
    non_existent_file = tmp_path / "non_existent.txt"
    with pytest.raises(FileNotFoundError):
        doc_processor.extract_text(non_existent_file)

def test_extract_text_unsupported_type(doc_processor, temp_unsupported_file):
    """Tests handling of an unsupported file type."""
    file_path = temp_unsupported_file
    with pytest.raises(ValueError, match="Unsupported file type"):
        doc_processor.extract_text(file_path)

