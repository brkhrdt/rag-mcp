import pytest
import subprocess
from pathlib import Path
import shutil

# Define the path to the main CLI script
CLI_SCRIPT = Path(__file__).parent.parent / "src" / "rag_mcp" / "main.py"


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


def run_cli_command(command: list[str], db_path: Path):
    """
    Helper function to run a CLI command and return its stdout, stderr, and return code.
    """
    full_command = [
        ".venv/bin/ragmcp",
        "--db-path",
        str(db_path),
    ] + command
    result = subprocess.run(full_command, capture_output=True, text=True, check=False)
    return result.stdout, result.stderr, result.returncode


def test_cli_ingest_file(temp_chroma_dir, temp_ingest_file):
    """
    Test ingesting a file via the CLI.
    """
    stdout, stderr, returncode = run_cli_command(
        ["ingest-file", str(temp_ingest_file)], temp_chroma_dir
    )

    assert returncode == 0, f"CLI exited with error: {stderr}"
    # assert not stdout, f"{stdout}"
    assert "Batches:" in stderr

    # Verify that the document was ingested by querying the RAG system directly
    # (This requires importing RAG, but it's for verification, not part of the CLI test itself)
    from rag_mcp.rag import RAG

    rag_system = RAG(chroma_persist_directory=str(temp_chroma_dir))
    results = rag_system.query("test document", num_results=1)

    assert len(results) > 0
    assert "test document" in results[0].document.lower()


def test_cli_ingest_file_with_tags(temp_chroma_dir, temp_ingest_file):
    """
    Test ingesting a file via the CLI with tags.
    """
    stdout, stderr, returncode = run_cli_command(
        ["ingest-file", str(temp_ingest_file), "--tags", "report", "Q1"],
        temp_chroma_dir,
    )

    assert returncode == 0, f"CLI exited with error: {stderr}"
    # assert not stdout
    assert "Batches:" in stderr

    from rag_mcp.rag import RAG

    rag_system = RAG(chroma_persist_directory=str(temp_chroma_dir))
    results = rag_system.query("test document", num_results=1)

    assert len(results) > 0
    assert "report" in results[0].metadata["tags"]
    assert "Q1" in results[0].metadata["tags"]


def test_cli_ingest_text(temp_chroma_dir):
    """
    Test ingesting a text string via the CLI.
    """
    test_text = "This is a direct text input for testing."
    stdout, stderr, returncode = run_cli_command(
        ["ingest-text", test_text, "--source-name", "cli_input"], temp_chroma_dir
    )

    assert returncode == 0, f"CLI exited with error: {stderr}"
    # assert not stdout
    assert "Batches:" in stderr

    from rag_mcp.rag import RAG

    rag_system = RAG(chroma_persist_directory=str(temp_chroma_dir))
    results = rag_system.query("direct text input", num_results=1)

    assert len(results) > 0
    assert "direct text input" in results[0].document.lower()
    assert results[0].metadata["source"] == "cli_input"


def test_cli_query(temp_chroma_dir, temp_ingest_file):
    """
    Test querying via the CLI after ingestion.
    """
    # First, ingest a file
    ingest_stdout, ingest_stderr, ingest_returncode = run_cli_command(
        ["ingest-file", str(temp_ingest_file)], temp_chroma_dir
    )
    assert ingest_returncode == 0, f"Ingestion failed: {ingest_stderr}"

    # Then, query
    query_stdout, query_stderr, query_returncode = run_cli_command(
        ["query", "information"], temp_chroma_dir
    )

    assert query_returncode == 0, f"CLI exited with error: {query_stderr}"
    assert "--- Query Results ---" in query_stdout
    assert "test document" in query_stdout.lower()
    assert not query_stderr


def test_cli_no_command(temp_chroma_dir):
    """
    Test running the CLI with no command, which should print help and exit with error.
    """
    stdout, stderr, returncode = run_cli_command([], temp_chroma_dir)

    assert returncode == 1  # Expecting an error exit code
    assert "usage: ragmcp" in stdout or "usage: ragmcp" in stderr
    assert "Available commands" in stdout or "Available commands" in stderr


def test_cli_query_no_results(temp_chroma_dir):
    """
    Test querying via the CLI when no results are found.
    """
    stdout, stderr, returncode = run_cli_command(
        ["query", "nonexistent query"], temp_chroma_dir
    )

    assert returncode == 0  # Successful execution, just no results
    assert "No results found for your query." in stdout

    assert not stderr


def test_cli_ingest_file_glob_pattern(temp_chroma_dir, tmp_path):
    """
    Test ingesting files using a glob pattern.
    """
    file1_path = tmp_path / "doc1.txt"
    file1_path.write_text("Content of document one.")
    file2_path = tmp_path / "doc2.txt"
    file2_path.write_text("Content of document two.")

    stdout, stderr, returncode = run_cli_command(
        ["ingest-file", str(tmp_path / "*.txt")], temp_chroma_dir
    )

    assert returncode == 0, f"CLI exited with error: {stderr}"
    # assert not stdout
    assert "Batches:" in stderr

    from rag_mcp.rag import RAG

    rag_system = RAG(chroma_persist_directory=str(temp_chroma_dir))
    results1 = rag_system.query("document one", num_results=1)
    results2 = rag_system.query("document two", num_results=1)

    assert len(results1) > 0
    assert "document one" in results1[0].document.lower()
    assert len(results2) > 0
    assert "document two" in results2[0].document.lower()
