import pytest
from rag_mcp.rag import RAG
import shutil


@pytest.fixture
def rag_system_temp(tmp_path):
    """Provides a RAG instance with a temporary ChromaDB directory."""
    chroma_persist_dir = tmp_path / "test_chroma_db_rag_system"

    if chroma_persist_dir.exists():
        shutil.rmtree(chroma_persist_dir)

    system = RAG(
        chroma_collection_name="test_rag_collection",
        chroma_persist_directory=str(chroma_persist_dir),
    )
    yield system

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
    """Tests the ingestion process of the RAG."""
    file_path, content = temp_ingest_file
    rag_system_temp.ingest_file(file_path, chunk_size=10, chunk_overlap=2)

    query_results = rag_system_temp.query("fox jumps", num_results=1)
    assert len(query_results) > 0
    assert "fox jumps over the lazy dog" in query_results[0]["document"]
    assert query_results[0]["metadata"]["source"] == str(file_path)
    assert "chunk_index" in query_results[0]["metadata"]


def test_ingest_string_input(rag_system_temp):
    """Tests the ingestion process with a direct string input."""
    test_string = "This is a test string that will be ingested directly. It contains some unique words."
    source_name = "my_custom_string_source"
    rag_system_temp.ingest_string(
        test_string, chunk_size=10, chunk_overlap=2, source_name=source_name
    )

    query_results = rag_system_temp.query("unique words", num_results=1)
    assert len(query_results) > 0
    assert "unique words" in query_results[0]["document"]
    assert query_results[0]["metadata"]["source"] == source_name
    assert "chunk_index" in query_results[0]["metadata"]

    # Reset the vector store to ensure a clean state for the next test case
    rag_system_temp.reset_vector_store()

    # Test with default source name
    test_string_default = "Another string for testing default source name."
    rag_system_temp.ingest_string(test_string_default, chunk_size=10, chunk_overlap=2)
    query_results_default = rag_system_temp.query("default source", num_results=1)
    assert len(query_results_default) > 0
    assert query_results_default[0]["metadata"]["source"] == "string_input"


def test_query_system(rag_system_temp, temp_ingest_file):
    """Tests the query functionality of the RAG."""
    file_path, content = temp_ingest_file
    rag_system_temp.ingest_file(file_path, chunk_size=10, chunk_overlap=2)

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

    rag_system_temp.ingest_file(unsupported_file)

    captured = capsys.readouterr()
    assert "Error ingesting" in captured.out
    assert "Unsupported file type" in captured.out


def test_reset_vector_store(rag_system_temp, temp_ingest_file):
    """Tests resetting the vector store via RAG."""
    file_path, content = temp_ingest_file
    rag_system_temp.ingest_file(file_path)

    query_results_before_reset = rag_system_temp.query("test sentence", num_results=1)
    assert len(query_results_before_reset) > 0

    rag_system_temp.reset_vector_store()

    query_results_after_reset = rag_system_temp.query("test sentence", num_results=1)
    assert len(query_results_after_reset) == 0


def test_ingest_large_chunk_size_warning(rag_system_temp, temp_ingest_file, capsys):
    """
    Tests that a warning is issued and chunk_size is adjusted when it exceeds
    the embedding model's max_input_tokens.
    """
    file_path, content = temp_ingest_file

    large_chunk_size = 2000
    rag_system_temp.ingest_file(
        file_path, chunk_size=large_chunk_size, chunk_overlap=50
    )

    captured = capsys.readouterr()

    assert (
        f"Warning: Requested chunk_size ({large_chunk_size}) exceeds embedding model's max input tokens ({rag_system_temp.embedding_model.max_input_tokens})."
        in captured.out
    )
    assert (
        f"Using effective_chunk_size of {rag_system_temp.embedding_model.max_input_tokens}."
        in captured.out
    )

    query_results = rag_system_temp.query("fox jumps", num_results=1)
    assert len(query_results) > 0
    assert "fox jumps over the lazy dog" in query_results[0]["document"]
