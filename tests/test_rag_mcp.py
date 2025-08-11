import pytest
from unittest.mock import patch, MagicMock

# Import the functions from rag_mcp.py that are exposed as MCP tools
from rag_mcp.rag_mcp import ingest_file, ingest_text, query, reset_vector_store


@pytest.fixture
def mock_rag_system_mcp():
    """
    Fixture to mock the RAG system within the rag_mcp module.
    This ensures that actual RAG operations (like ChromaDB interactions) are not performed during tests.
    """
    with patch("rag_mcp.rag_mcp.RAG") as mock_rag_class:
        # Configure the mock instance that rag_mcp.rag_mcp.rag_system will be
        instance = mock_rag_class.return_value
        instance.ingest_file = MagicMock()
        instance.ingest_string = MagicMock()
        instance.query = MagicMock(return_value=[])
        instance.reset_vector_store = MagicMock()
        yield instance  # Yield the mock instance for direct interaction in tests


@pytest.mark.asyncio
async def test_ingest_file_mcp_tool(mock_rag_system_mcp, tmp_path):
    """
    Test the ingest_file MCP tool with a single file.
    """
    test_file = tmp_path / "doc1.txt"
    test_file.write_text("Content of document 1.")

    result = await ingest_file(
        file_paths=[str(test_file)],
        chunk_size=100,
        chunk_overlap=20,
        tags=["test_tag"],
    )

    mock_rag_system_mcp.ingest_file.assert_called_once_with(
        test_file, chunk_size=100, chunk_overlap=20, tags=["test_tag"]
    )
    assert "Successfully ingested 1 file(s)." in result
    assert str(test_file) in result


@pytest.mark.asyncio
async def test_ingest_file_mcp_tool_glob(mock_rag_system_mcp, tmp_path):
    """
    Test the ingest_file MCP tool with a glob pattern.
    """
    test_file1 = tmp_path / "doc_a.txt"
    test_file1.write_text("Content A.")
    test_file2 = tmp_path / "doc_b.txt"
    test_file2.write_text("Content B.")
    # Create a non-matching file to ensure glob works correctly
    tmp_path / "other.md"

    result = await ingest_file(file_paths=[str(tmp_path / "*.txt")])

    assert mock_rag_system_mcp.ingest_file.call_count == 2
    mock_rag_system_mcp.ingest_file.assert_any_call(
        test_file1, chunk_size=None, chunk_overlap=50, tags=None
    )
    mock_rag_system_mcp.ingest_file.assert_any_call(
        test_file2, chunk_size=None, chunk_overlap=50, tags=None
    )
    assert "Successfully ingested 2 file(s)." in result
    assert str(test_file1) in result
    assert str(test_file2) in result


@pytest.mark.asyncio
async def test_ingest_file_mcp_tool_non_existent_file(mock_rag_system_mcp, tmp_path):
    """
    Test the ingest_file MCP tool with a non-existent file.
    """
    non_existent_file = tmp_path / "non_existent.txt"
    result = await ingest_file(file_paths=[str(non_existent_file)])

    mock_rag_system_mcp.ingest_file.assert_not_called()
    assert "Successfully ingested 0 file(s)." in result
    assert f"{non_existent_file} (Skipped: Not a file)" in result


@pytest.mark.asyncio
async def test_ingest_file_mcp_tool_ingestion_error(mock_rag_system_mcp, tmp_path):
    """
    Test the ingest_file MCP tool when an ingestion error occurs.
    """
    test_file = tmp_path / "error_doc.txt"
    test_file.write_text("This file will cause an error.")

    mock_rag_system_mcp.ingest_file.side_effect = Exception("Simulated ingestion error")

    result = await ingest_file(file_paths=[str(test_file)])

    mock_rag_system_mcp.ingest_file.assert_called_once_with(
        test_file, chunk_size=None, chunk_overlap=50, tags=None
    )
    assert "Successfully ingested 0 file(s)." in result
    assert f"{test_file} (Error: Simulated ingestion error)" in result


@pytest.mark.asyncio
async def test_ingest_text_mcp_tool(mock_rag_system_mcp):
    """
    Test the ingest_text MCP tool.
    """
    test_content = "This is a test text string."
    test_source = "my_custom_source"
    test_tags = ["tag1", "tag2"]

    result = await ingest_text(
        text_content=test_content,
        chunk_size=50,
        chunk_overlap=10,
        source_name=test_source,
        tags=test_tags,
    )

    mock_rag_system_mcp.ingest_string.assert_called_once_with(
        test_content,
        chunk_size=50,
        chunk_overlap=10,
        source_name=test_source,
        tags=test_tags,
    )
    assert result == "Text content successfully ingested."


@pytest.mark.asyncio
async def test_ingest_text_mcp_tool_error(mock_rag_system_mcp):
    """
    Test the ingest_text MCP tool when an error occurs during ingestion.
    """
    mock_rag_system_mcp.ingest_string.side_effect = Exception("Text ingestion failed")

    result = await ingest_text(text_content="Some text.")

    assert "Failed to ingest text: Text ingestion failed" in result


@pytest.mark.asyncio
async def test_query_mcp_tool(mock_rag_system_mcp):
    """
    Test the query MCP tool with results.
    """
    mock_rag_system_mcp.query.return_value = [
        {
            "document": "Doc 1 content.",
            "distance": 0.05,
            "metadata": {
                "source": "file_a.txt",
                "chunk_index": 0,
                "timestamp": "2023-01-01T10:00:00",
                "tags": ["important", "finance"],
            },
        },
        {
            "document": "Doc 2 content.",
            "distance": 0.15,
            "metadata": {
                "source": "file_b.txt",
                "chunk_index": 1,
                "timestamp": "2023-01-01T11:00:00",
            },
        },
    ]

    results = await query(query_text="test query", num_results=2)

    mock_rag_system_mcp.query.assert_called_once_with("test query", 2)
    assert len(results) == 2

    assert results[0]["result_number"] == 1
    assert results[0]["source"] == "file_a.txt"
    assert results[0]["chunk_index"] == 0
    assert results[0]["timestamp"] == "2023-01-01T10:00:00"
    assert results[0]["distance"] == "0.0500"
    assert results[0]["document"] == "Doc 1 content."
    assert results[0]["tags"] == ["important", "finance"]

    assert results[1]["result_number"] == 2
    assert results[1]["source"] == "file_b.txt"
    assert results[1]["chunk_index"] == 1
    assert results[1]["timestamp"] == "2023-01-01T11:00:00"
    assert results[1]["distance"] == "0.1500"
    assert results[1]["document"] == "Doc 2 content."
    assert "tags" not in results[1]  # Ensure tags key is not present if not in metadata


@pytest.mark.asyncio
async def test_query_mcp_tool_no_results(mock_rag_system_mcp, caplog):
    """
    Test the query MCP tool when no results are found.
    """
    mock_rag_system_mcp.query.return_value = []

    with caplog.at_level("INFO"):
        results = await query(query_text="no match")

    mock_rag_system_mcp.query.assert_called_once_with("no match", 5)
    assert results == []
    assert "No results found for your query." in caplog.text


@pytest.mark.asyncio
async def test_query_mcp_tool_error(mock_rag_system_mcp):
    """
    Test the query MCP tool when an error occurs during the query.
    """
    mock_rag_system_mcp.query.side_effect = Exception("Query failed")

    results = await query(query_text="error query")

    assert len(results) == 1
    assert "error" in results[0]
    assert "Failed to perform query: Query failed" in results[0]["error"]


@pytest.mark.asyncio
async def test_reset_vector_store_mcp_tool(mock_rag_system_mcp):
    """
    Test the reset_vector_store MCP tool.
    """
    result = await reset_vector_store()

    mock_rag_system_mcp.reset_vector_store.assert_called_once()
    assert result == "Vector store has been successfully reset."


@pytest.mark.asyncio
async def test_reset_vector_store_mcp_tool_error(mock_rag_system_mcp):
    """
    Test the reset_vector_store MCP tool when an error occurs.
    """
    mock_rag_system_mcp.reset_vector_store.side_effect = Exception("Reset failed")

    result = await reset_vector_store()

    assert "Failed to reset vector store: Reset failed" in result
