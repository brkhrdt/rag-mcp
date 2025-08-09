# tests/test_rag_system.py
import pytest
from rag_mcp.rag_system import RAGSystem
import shutil


# Fixture for a temporary RAGSystem instance
@pytest.fixture
def rag_system_temp(tmp_path):
    """Provides a RAGSystem instance with a temporary ChromaDB directory."""
    chroma_persist_dir = tmp_path / "test_chroma_db_rag_system"
    # Ensure the directory is clean before starting
    if chroma_persist_dir.exists():
        shutil.rmtree(chroma_persist_dir)

    system = RAGSystem(
        chroma_collection_name="test_rag_collection",
        chroma_persist_directory=str(chroma_persist_dir),
    )
    yield system
    # Teardown: Clean up the ChromaDB directory after tests
    if chroma_persist_dir.exists():
        shutil.rmtree(chroma_persist_dir)


@pytest.fixture
def temp_ingest_file(tmp_path):
    """Creates a temporary text file for ingestion tests."""
    content = (
        "The quick brown fox jumps over the lazy dog. "
        "This is a test sentence for chunking and embedding. "
        "Artificial intelligence is a rapidly developing field."
    )
    file_path = tmp_path / "ingest_doc.txt"
    file_path.write_text(content)
    return file_path, content


def test_ingest_document(rag_system_temp, temp_ingest_file):
    """Tests the ingestion process of the RAGSystem."""
    file_path, content = temp_ingest_file
    rag_system_temp.ingest(file_path, chunk_size=10, chunk_overlap=2)

    # Verify that documents were added to the vector store
    # We can't directly access the count from the fixture, but we can query
    # for a known part of the document and expect results.
    # A more direct check would involve mocking or accessing internal state,
    # but querying is a good integration test.
    query_results = rag_system_temp.query("fox jumps", num_results=1)
    assert len(query_results) > 0
    assert "fox jumps over the lazy dog" in query_results[0]["document"]
    assert query_results[0]["metadata"]["source"] == str(file_path)
    assert "chunk_index" in query_results[0]["metadata"]


def test_query_system(rag_system_temp, temp_ingest_file):
    """Tests the query functionality of the RAGSystem."""
    file_path, content = temp_ingest_file
    rag_system_temp.ingest(file_path, chunk_size=10, chunk_overlap=2)

    query_text = "What is AI?"
    results = rag_system_temp.query(query_text, num_results=1)

    assert len(results) == 1
    assert "Artificial intelligence" in results[0]["document"]
    assert results[0]["metadata"]["source"] == str(file_path)
    assert "distance" in results[0]


def test_ingest_unsupported_file_type(rag_system_temp, tmp_path, capsys):
    """Tests ingestion of an unsupported file type."""
    unsupported_file = tmp_path / "unsupported.pdf"
    unsupported_file.write_text("dummy pdf content")

    rag_system_temp.ingest(unsupported_file)

    # Check if the error message was printed
    captured = capsys.readouterr()
    assert "Error ingesting" in captured.out
    assert "Unsupported file type" in captured.out


def test_reset_vector_store(rag_system_temp, temp_ingest_file):
    """Tests resetting the vector store via RAGSystem."""
    file_path, content = temp_ingest_file
    rag_system_temp.ingest(file_path)

    # Verify content exists
    query_results_before_reset = rag_system_temp.query("test sentence", num_results=1)
    assert len(query_results_before_reset) > 0

    rag_system_temp.reset_vector_store()

    # Verify content is gone after reset
    query_results_after_reset = rag_system_temp.query("test sentence", num_results=1)
    assert len(query_results_after_reset) == 0
