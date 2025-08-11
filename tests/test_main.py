import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from rag_mcp.main import main
from rag_mcp.rag import RAG


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_rag_system():
    with patch("rag_mcp.main.RAG") as mock_rag:
        instance = mock_rag.return_value
        instance.ingest_file = MagicMock()
        instance.ingest_string = MagicMock()
        instance.query = MagicMock(return_value=[])
        yield instance


def test_ingest_file_command(runner, mock_rag_system, tmp_path):
    # Create a dummy file for ingestion
    test_file = tmp_path / "test_doc.txt"
    test_file.write_text("This is a test document.")

    result = runner.invoke(
        main,
        [
            "ingest-file",
            str(test_file),
            "--chunk-size",
            "100",
            "--chunk-overlap",
            "20",
            "--tags",
            "test_tag1",
            "test_tag2",
        ],
    )

    assert result.exit_code == 0
    mock_rag_system.ingest_file.assert_called_once_with(
        test_file, chunk_size=100, chunk_overlap=20, tags=["test_tag1", "test_tag2"]
    )


def test_ingest_file_command_glob(runner, mock_rag_system, tmp_path):
    # Create dummy files for ingestion
    test_file1 = tmp_path / "test_doc1.txt"
    test_file1.write_text("This is test document 1.")
    test_file2 = tmp_path / "test_doc2.txt"
    test_file2.write_text("This is test document 2.")

    result = runner.invoke(
        main,
        [
            "ingest-file",
            str(tmp_path / "*.txt"),
        ],
    )

    assert result.exit_code == 0
    assert mock_rag_system.ingest_file.call_count == 2
    # Check if both files were called, order might vary
    mock_rag_system.ingest_file.assert_any_call(
        test_file1, chunk_size=None, chunk_overlap=50, tags=None
    )
    mock_rag_system.ingest_file.assert_any_call(
        test_file2, chunk_size=None, chunk_overlap=50, tags=None
    )


def test_ingest_text_command(runner, mock_rag_system):
    test_text = "This is some test text."
    result = runner.invoke(
        main,
        [
            "ingest-text",
            test_text,
            "--chunk-size",
            "50",
            "--chunk-overlap",
            "10",
            "--source-name",
            "my_source",
            "--tags",
            "tag_a",
            "tag_b",
        ],
    )

    assert result.exit_code == 0
    mock_rag_system.ingest_string.assert_called_once_with(
        test_text,
        chunk_size=50,
        chunk_overlap=10,
        source_name="my_source",
        tags=["tag_a", "tag_b"],
    )


def test_query_command(runner, mock_rag_system):
    mock_rag_system.query.return_value = [
        {
            "document": "Result 1 content.",
            "distance": 0.1,
            "metadata": {
                "source": "file1.txt",
                "chunk_index": 0,
                "timestamp": "2023-01-01",
                "tags": ["tag1"],
            },
        },
        {
            "document": "Result 2 content.",
            "distance": 0.2,
            "metadata": {
                "source": "file2.txt",
                "chunk_index": 1,
                "timestamp": "2023-01-02",
            },
        },
    ]

    result = runner.invoke(main, ["query", "What is the capital of France?", "--num-results", "2"])

    assert result.exit_code == 0
    mock_rag_system.query.assert_called_once_with("What is the capital of France?", 2)
    assert "Result 1:" in result.output
    assert "Source: file1.txt" in result.output
    assert "Chunk Index: 0" in result.output
    assert "Timestamp: 2023-01-01" in result.output
    assert "Tags: tag1" in result.output
    assert "Distance: 0.1000" in result.output
    assert "Document:\nResult 1 content." in result.output
    assert "Result 2:" in result.output
    assert "Source: file2.txt" in result.output
    assert "Tags: N/A" not in result.output # Ensure tags line is not printed if not present
    assert "Document:\nResult 2 content." in result.output


def test_query_command_no_results(runner, mock_rag_system):
    mock_rag_system.query.return_value = []
    result = runner.invoke(main, ["query", "empty query"])
    assert result.exit_code == 0
    assert "No results found for your query." in result.output


def test_main_no_command(runner):
    result = runner.invoke(main, [])
    assert result.exit_code != 0  # Should exit with an error for no command
    assert "Error: Missing command" in result.output or "Usage:" in result.output


def test_db_path_argument(runner, mock_rag_system, tmp_path):
    test_file = tmp_path / "test_doc.txt"
    test_file.write_text("Content.")

    db_path = str(tmp_path / "my_custom_db")
    result = runner.invoke(main, ["--db-path", db_path, "ingest-file", str(test_file)])

    assert result.exit_code == 0
    # Verify that RAG was initialized with the custom db_path
    RAG.assert_called_once_with(chroma_persist_directory=db_path)
