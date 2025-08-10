import pytest
from rag_mcp.document_processor import DocumentProcessor


@pytest.fixture
def doc_processor():
    """Fixture for DocumentProcessor."""
    return DocumentProcessor()


@pytest.fixture
def temp_txt_file(tmp_path):
    """Fixture for a temporary .txt file."""
    file_content = "This is a test text file."
    file_path = tmp_path / "test_doc.txt"
    file_path.write_text(file_content)
    return file_path


@pytest.fixture
def temp_binary_file(tmp_path):
    """Creates a temporary binary file for testing."""
    file_path = tmp_path / "test_binary.bin"
    # Write some non-UTF-8 compliant bytes
    file_path.write_bytes(b"\x80\x01\x02\x03")
    return file_path


def test_extract_text_txt(doc_processor, temp_txt_file):
    """Tests text extraction from a .txt file."""
    extracted_text = doc_processor.extract_text(temp_txt_file)
    assert extracted_text == "This is a test text file."


def test_extract_text_file_not_found(doc_processor, tmp_path):
    """Tests handling of a non-existent file."""
    non_existent_file = tmp_path / "non_existent.txt"
    with pytest.raises(FileNotFoundError, match="File not found"):
        doc_processor.extract_text(non_existent_file)


def test_extract_text_binary_file_raises_unicode_decode_error(
    doc_processor, temp_binary_file
):
    """Tests that extracting text from a binary file raises UnicodeDecodeError."""
    with pytest.raises(UnicodeDecodeError):
        doc_processor.extract_text(temp_binary_file)
