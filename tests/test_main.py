import pytest
import sys
import io
from unittest.mock import patch, MagicMock
from rag_mcp.main import main
from rag_mcp.rag import RAG


@pytest.fixture
def mock_rag_system():
    with patch("rag_mcp.main.RAG") as mock_rag:
        instance = mock_rag.return_value
        instance.ingest_file = MagicMock()
        instance.ingest_string = MagicMock()
        instance.query = MagicMock(return_value=[])
        yield mock_rag, instance # Yield both the mock class and the instance


def test_ingest_file_command(mock_rag_system, tmp_path):
    mock_rag_class, _ = mock_rag_system
    # Create a dummy file for ingestion
    test_file = tmp_path / "test_doc.txt"
    test_file.write_text("This is a test document.")

    test_args = [
        "main.py",  # sys.argv[0] is typically the script name
        "ingest-file",
        str(test_file),
        "--chunk-size",
        "100",
        "--chunk-overlap",
        "20",
        "--tags",
        "test_tag1",
        "test_tag2",
    ]

    with patch.object(sys, "argv", test_args):
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            main()
            # No direct output expected for ingest commands, but check if it runs without error
            assert mock_stdout.getvalue() == ""

    mock_rag_class.return_value.ingest_file.assert_called_once_with(
        test_file, chunk_size=100, chunk_overlap=20, tags=["test_tag1", "test_tag2"]
    )


def test_ingest_file_command_glob(mock_rag_system, tmp_path):
    mock_rag_class, _ = mock_rag_system
    # Create dummy files for ingestion
    test_file1 = tmp_path / "test_doc1.txt"
    test_file1.write_text("This is test document 1.")
    test_file2 = tmp_path / "test_doc2.txt"
    test_file2.write_text("This is test document 2.")

    test_args = [
        "main.py",
        "ingest-file",
        str(tmp_path / "*.txt"),
    ]

    with patch.object(sys, "argv", test_args):
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            main()
            assert mock_stdout.getvalue() == ""

    assert mock_rag_class.return_value.ingest_file.call_count == 2
    # Check if both files were called, order might vary
    mock_rag_class.return_value.ingest_file.assert_any_call(
        test_file1, chunk_size=None, chunk_overlap=50, tags=None
    )
    mock_rag_class.return_value.ingest_file.assert_any_call(
        test_file2, chunk_size=None, chunk_overlap=50, tags=None
    )


def test_ingest_text_command(mock_rag_system):
    mock_rag_class, _ = mock_rag_system
    test_text = "This is some test text."
    test_args = [
        "main.py",
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
    ]

    with patch.object(sys, "argv", test_args):
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            main()
            assert mock_stdout.getvalue() == ""

    mock_rag_class.return_value.ingest_string.assert_called_once_with(
        test_text,
        chunk_size=50,
        chunk_overlap=10,
        source_name="my_source",
        tags=["tag_a", "tag_b"],
    )


def test_query_command(mock_rag_system):
    mock_rag_class, _ = mock_rag_system
    mock_rag_class.return_value.query.return_value = [
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

    test_args = [
        "main.py",
        "query",
        "What is the capital of France?",
        "--num-results",
        "2",
    ]

    with patch.object(sys, "argv", test_args):
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            main()
            output = mock_stdout.getvalue()

    mock_rag_class.return_value.query.assert_called_once_with("What is the capital of France?", 2)
    assert "Result 1:" in output
    assert "Source: file1.txt" in output
    assert "Chunk Index: 0" in output
    assert "Timestamp: 2023-01-01" in output
    assert "Tags: tag1" in output
    assert "Distance: 0.1000" in output
    assert "Document:\nResult 1 content." in output
    assert "Result 2:" in output
    assert "Source: file2.txt" in output
    assert "Tags: N/A" not in output  # Ensure tags line is not printed if not present
    assert "Document:\nResult 2 content." in output


def test_query_command_no_results(mock_rag_system):
    mock_rag_class, _ = mock_rag_system
    mock_rag_class.return_value.query.return_value = []
    test_args = ["main.py", "query", "empty query"]

    with patch.object(sys, "argv", test_args):
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            main()
            output = mock_stdout.getvalue()

    assert "No results found for your query." in output


def test_main_no_command():
    test_args = ["main.py"]  # No command provided

    with patch.object(sys, "argv", test_args):
        # Patch parser.error to raise SystemExit instead of calling sys.exit
        with patch("argparse.ArgumentParser.error", side_effect=SystemExit) as mock_error:
            with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
                with pytest.raises(SystemExit) as excinfo:
                    main()
                assert excinfo.value.code != 0  # Should exit with a non-zero code
                # Check for expected error message in stderr
                assert "usage:" in mock_stderr.getvalue().lower()
                assert (
                    "the following arguments are required: command"
                    in mock_stderr.getvalue().lower()
                )
                mock_error.assert_called_once() # Ensure error was called


def test_db_path_argument(mock_rag_system, tmp_path):
    mock_rag_class, _ = mock_rag_system
    test_file = tmp_path / "test_doc.txt"
    test_file.write_text("Content.")

    db_path = str(tmp_path / "my_custom_db")
    test_args = ["main.py", "--db-path", db_path, "ingest-file", str(test_file)]

    with patch.object(sys, "argv", test_args):
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            main()
            assert mock_stdout.getvalue() == ""

    # Verify that RAG was initialized with the custom db_path
    mock_rag_class.assert_called_once_with(chroma_persist_directory=db_path)

