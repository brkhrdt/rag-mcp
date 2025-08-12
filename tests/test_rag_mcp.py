import pytest
import os
from typing import Dict, List

# Import the FastMCP client for testing
from mcp.server.fastmcp import FastMCP

# Import the module under test
import rag_mcp.rag_mcp as rag_mcp_module

# Import the RAG class for direct verification
from rag_mcp.rag import RAG


# Re-use fixtures from test_main.py
# If test_main.py is not in the same directory or accessible,
# these fixtures would need to be defined here or in a conftest.py
@pytest.fixture
def temp_chroma_dir(tmp_path):
    """
    Fixture to create a temporary directory for ChromaDB and clean it up afterwards.
    """
    db_path = tmp_path / "chroma_test_db"
    db_path.mkdir()
    yield db_path
    # Clean up the directory after the test
    if db_path.exists():
        import shutil

        shutil.rmtree(db_path)


@pytest.fixture
def temp_ingest_file(tmp_path):
    """
    Fixture to create a temporary text file for ingestion tests.
    """
    file_content = "This is a test document. It contains some information."
    file_path = tmp_path / "test_document.txt"
    file_path.write_text(file_content)
    return file_path


@pytest.fixture
async def mcp_client():
    """
    Fixture to provide a FastMCP test client.
    """
    # Initialize FastMCP with a dummy name for testing
    test_mcp = FastMCP("test-rag-mcp")
    # Register the tools from the rag_mcp_module
    # This is crucial because the tools are defined using @mcp.tool() decorator
    # in the rag_mcp_module, and we need to ensure they are registered with *this*
    # FastMCP instance for testing.
    # However, since rag_mcp_module already initializes its own global mcp instance,
    # we will use that one directly for testing its tools.
    # The FastMCP client is used to call the registered tools.
    yield rag_mcp_module.mcp.client()


@pytest.fixture(autouse=True)
async def rag_system_fixture(temp_chroma_dir):
    """
    Fixture to manage the global rag_system instance in rag_mcp.py for tests.
    It ensures each test gets a clean RAG instance pointing to a temporary DB.
    """
    original_rag_system = rag_mcp_module.rag_system
    original_db_path = os.environ.get("RAG_CHROMADB_PATH")

    # Set the environment variable for the RAG system to use the temporary path
    os.environ["RAG_CHROMADB_PATH"] = str(temp_chroma_dir)

    # Force re-initialization of the RAG system in the module
    # by setting it to None, so _get_rag_system() creates a new one
    rag_mcp_module.rag_system = None

    yield

    # Clean up: Reset the global rag_system and environment variable
    rag_mcp_module.rag_system = original_rag_system
    if original_db_path is not None:
        os.environ["RAG_CHROMADB_PATH"] = original_db_path
    else:
        # Only delete if it was not set originally
        if "RAG_CHROMADB_PATH" in os.environ:
            del os.environ["RAG_CHROMADB_PATH"]


@pytest.mark.asyncio
async def test_ingest_file(mcp_client, temp_ingest_file, temp_chroma_dir):
    """
    Test the ingest_file tool.
    """
    file_path_str = str(temp_ingest_file)
    response = await mcp_client.ingest_file(file_paths=[file_path_str])

    assert "Ingested 1 file(s)." in response
    assert file_path_str in response

    # Verify ingestion by querying directly via RAG (not mocking)
    rag_instance = RAG(chroma_persist_directory=str(temp_chroma_dir))
    results = rag_instance.query("test document", num_results=1)
    assert len(results) > 0
    assert "test document" in results[0].document.lower()


@pytest.mark.asyncio
async def test_ingest_file_with_tags(mcp_client, temp_ingest_file, temp_chroma_dir):
    """
    Test the ingest_file tool with tags.
    """
    file_path_str = str(temp_ingest_file)
    tags = ["report", "Q1"]
    response = await mcp_client.ingest_file(file_paths=[file_path_str], tags=tags)

    assert "Ingested 1 file(s)." in response
    assert file_path_str in response

    # Verify ingestion and tags
    rag_instance = RAG(chroma_persist_directory=str(temp_chroma_dir))
    results = rag_instance.query("test document", num_results=1)
    assert len(results) > 0
    assert "test document" in results[0].document.lower()
    assert "tags" in results[0].metadata
    assert set(tags) == set(results[0].metadata["tags"])


@pytest.mark.asyncio
async def test_ingest_text(mcp_client, temp_chroma_dir):
    """
    Test the ingest_text tool.
    """
    test_text = "This is a direct text input for testing the ingest_text tool."
    source_name = "cli_test_input"
    response = await mcp_client.ingest_text(
        text_content=test_text, source_name=source_name
    )

    assert "Text content successfully ingested." in response

    # Verify ingestion
    rag_instance = RAG(chroma_persist_directory=str(temp_chroma_dir))
    results = rag_instance.query("direct text input", num_results=1)
    assert len(results) > 0
    assert "direct text input" in results[0].document.lower()
    assert results[0].metadata["source"] == source_name


@pytest.mark.asyncio
async def test_query(mcp_client, temp_ingest_file, temp_chroma_dir):
    """
    Test the query tool after ingesting a file.
    """
    # First, ingest a file
    await mcp_client.ingest_file(file_paths=[str(temp_ingest_file)])

    # Then, query
    query_results: List[Dict] = await mcp_client.query(query_text="information")

    assert isinstance(query_results, list)
    assert len(query_results) > 0
    assert "document" in query_results[0]
    assert "metadata" in query_results[0]
    assert "distance" in query_results[0]
    assert "test document" in query_results[0]["document"].lower()


@pytest.mark.asyncio
async def test_query_no_results(mcp_client):
    """
    Test the query tool when no results are found (empty vector store).
    """
    query_results: List[Dict] = await mcp_client.query(query_text="nonexistent query")

    assert isinstance(query_results, list)
    assert len(query_results) == 0


@pytest.mark.asyncio
async def test_reset_vector_store(mcp_client, temp_ingest_file, temp_chroma_dir):
    """
    Test the reset_vector_store tool.
    """
    # Ingest a file first to ensure there's something to reset
    await mcp_client.ingest_file(file_paths=[str(temp_ingest_file)])

    # Verify content exists before reset
    rag_instance_before_reset = RAG(chroma_persist_directory=str(temp_chroma_dir))
    results_before_reset = rag_instance_before_reset.query(
        "test document", num_results=1
    )
    assert len(results_before_reset) > 0

    # Perform reset
    response = await mcp_client.reset_vector_store()
    assert "Vector store has been successfully reset." in response

    # Verify content is gone after reset
    rag_instance_after_reset = RAG(chroma_persist_directory=str(temp_chroma_dir))
    results_after_reset = rag_instance_after_reset.query("test document", num_results=1)
    assert len(results_after_reset) == 0
