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


def test_ingest_multiple_documents(rag_system_temp, tmp_path):
    """Tests ingesting multiple documents and querying across them."""
    content1 = "The first document talks about apples and oranges."
    file_path1 = tmp_path / "doc1.txt"
    file_path1.write_text(content1)

    content2 = "The second document discusses bananas and grapes."
    file_path2 = tmp_path / "doc2.txt"
    file_path2.write_text(content2)

    content3 = "The third document discusses lettuce."
    file_path3 = tmp_path / "doc3.txt"
    file_path3.write_text(content3)

    rag_system_temp.ingest_file(file_path1, chunk_size=10, chunk_overlap=2)
    rag_system_temp.ingest_file(file_path2, chunk_size=10, chunk_overlap=2)
    rag_system_temp.ingest_file(file_path3, chunk_size=10, chunk_overlap=2)

    # Query for content from the first document
    query_results1 = rag_system_temp.query("apples", num_results=1)
    assert len(query_results1) > 0
    assert "apples and oranges" in query_results1[0]["document"]
    assert query_results1[0]["metadata"]["source"] == str(file_path1)

    # Query for content from the second document
    query_results2 = rag_system_temp.query("bananas", num_results=1)
    assert len(query_results2) > 0
    assert "bananas and grapes" in query_results2[0]["document"]
    assert query_results2[0]["metadata"]["source"] == str(file_path2)

    # Query for content from the second document
    query_results3 = rag_system_temp.query("vegetable", num_results=1)
    assert len(query_results3) > 0
    assert "lettuce" in query_results3[0]["document"]
    assert query_results3[0]["metadata"]["source"] == str(file_path3)

    # Query for content that might span or be related to both (if applicable, though not strictly tested here)
    # For now, just ensure both types of content are retrievable
    query_results_combined = rag_system_temp.query("fruits", num_results=2)
    assert len(query_results_combined) == 2
    sources = {r["metadata"]["source"] for r in query_results_combined}
    assert str(file_path1) in sources
    assert str(file_path2) in sources


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


def test_reset_vector_store(rag_system_temp, temp_ingest_file):
    """Tests resetting the vector store via RAG."""
    file_path, content = temp_ingest_file
    rag_system_temp.ingest_file(file_path)

    query_results_before_reset = rag_system_temp.query("test sentence", num_results=1)
    assert len(query_results_before_reset) > 0

    rag_system_temp.reset_vector_store()

    query_results_after_reset = rag_system_temp.query("test sentence", num_results=1)
    assert len(query_results_after_reset) == 0


def test_ingest_large_chunk_size_warning(rag_system_temp, temp_ingest_file, caplog):
    """
    Tests that a warning is issued and chunk_size is adjusted when it exceeds
    the embedding model's max_input_tokens.
    """
    file_path, content = temp_ingest_file

    large_chunk_size = 2000

    # Set the logging level to WARNING to ensure the message is captured
    import logging

    caplog.set_level(logging.WARNING)

    rag_system_temp.ingest_file(
        file_path, chunk_size=large_chunk_size, chunk_overlap=50
    )

    expected_warning_message_part1 = f"Requested chunk_size ({large_chunk_size}) exceeds embedding model's max input tokens ({rag_system_temp.embedding_model.max_input_tokens})."
    expected_warning_message_part2 = f"Using effective_chunk_size of {rag_system_temp.embedding_model.max_input_tokens}."

    # Check if the warning message is present in the captured logs
    assert any(
        expected_warning_message_part1 in record.message for record in caplog.records
    )
    assert any(
        expected_warning_message_part2 in record.message for record in caplog.records
    )

    query_results = rag_system_temp.query("fox jumps", num_results=1)
    assert len(query_results) > 0
    assert "fox jumps over the lazy dog" in query_results[0]["document"]


def test_ingest_with_tags(rag_system_temp, temp_ingest_file):
    """Tests that documents are ingested with correct tags."""
    test_string = "This document is about finance and technology."
    test_tags = ["finance", "tech", "report"]
    source_name = "finance_tech_doc"

    rag_system_temp.ingest_string(
        test_string,
        chunk_size=10,
        chunk_overlap=2,
        source_name=source_name,
        tags=test_tags,
    )

    query_results = rag_system_temp.query("finance report", num_results=1)
    assert len(query_results) > 0
    assert query_results[0]["metadata"]["source"] == source_name
    assert "tags" in query_results[0]["metadata"]
    # Split the tags string back into a list for comparison
    retrieved_tags = query_results[0]["metadata"]["tags"].split(",")
    assert sorted(retrieved_tags) == sorted(test_tags)

    # Test file ingestion with tags
    file_path, content = temp_ingest_file
    file_tags = ["document", "test"]
    rag_system_temp.ingest_file(file_path, tags=file_tags)

    query_results_file = rag_system_temp.query("quick brown fox", num_results=1)
    assert len(query_results_file) > 0
    assert query_results_file[0]["metadata"]["source"] == str(file_path)
    assert "tags" in query_results_file[0]["metadata"]
    # Split the tags string back into a list for comparison
    retrieved_file_tags = query_results_file[0]["metadata"]["tags"].split(",")
    assert sorted(retrieved_file_tags) == sorted(file_tags)
