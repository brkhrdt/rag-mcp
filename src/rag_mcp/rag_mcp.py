from typing import Dict, Optional, List
from mcp.server.fastmcp import FastMCP
from pathlib import Path
import glob
import logging

from rag_mcp.rag import RAG

# Get a logger for this module
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("interactive-shell")

# Initialize RAG system (using default persistence directory for now)
# TODO: Make db_path configurable for MCP functions if needed
rag_system = RAG()


@mcp.tool()
async def ingest_file(
    file_paths: List[str],
    chunk_size: Optional[int] = None,
    chunk_overlap: int = 50,
    tags: Optional[List[str]] = None,
) -> str:
    """Ingest one or more document files into the RAG system.

    Args:
        file_paths: A list of paths to the document files to ingest (supports glob patterns).
        chunk_size: Maximum token size for each text chunk.
        chunk_overlap: Token overlap between chunks (default: 50).
        tags: Optional list of tags to associate with the ingested file(s).

    Returns:
        A message indicating the success or failure of the ingestion.
    """
    ingested_files = []
    skipped_files = []
    for pattern in file_paths:
        for file_path_str in glob.glob(pattern):
            file_path = Path(file_path_str)
            if file_path.is_file():
                logger.info(f"Ingesting file: {file_path}")
                try:
                    await rag_system.ingest_file(
                        file_path,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        tags=tags,
                    )
                    ingested_files.append(str(file_path))
                except Exception as e:
                    logger.error(f"Error ingesting {file_path}: {e}")
                    skipped_files.append(f"{file_path} (Error: {e})")
            else:
                logger.warning(f"Skipping non-file path: {file_path}")
                skipped_files.append(f"{file_path} (Skipped: Not a file)")

    response = (
        f"Ingestion complete. Successfully ingested {len(ingested_files)} file(s)."
    )
    if ingested_files:
        response += f"\nIngested: {', '.join(ingested_files)}"
    if skipped_files:
        response += f"\nSkipped/Failed: {', '.join(skipped_files)}"
    return response


@mcp.tool()
async def ingest_text(
    text_content: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: int = 50,
    source_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> str:
    """Ingest a text string directly into the RAG system.

    Args:
        text_content: The text string to ingest.
        chunk_size: Maximum token size for each text chunk.
        chunk_overlap: Token overlap between chunks (default: 50).
        source_name: Optional name for the source when ingesting a string (default: 'string_input').
        tags: Optional list of tags to associate with the ingested text.

    Returns:
        A message indicating the success or failure of the ingestion.
    """
    try:
        await rag_system.ingest_string(
            text_content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source_name=source_name,
            tags=tags,
        )
        return "Text content successfully ingested."
    except Exception as e:
        logger.error(f"Error ingesting text: {e}")
        return f"Failed to ingest text: {e}"


@mcp.tool()
async def query(
    query_text: str,
    num_results: int = 5,
) -> List[Dict]:
    """Query the RAG system for relevant information.

    Args:
        query_text: The query string.
        num_results: Number of top relevant results to retrieve (default: 5).

    Returns:
        A list of dictionaries, where each dictionary represents a query result
        containing 'document', 'metadata', and 'distance'.
    """
    try:
        results = await rag_system.query(query_text, num_results)
        if not results:
            logger.info("No results found for your query.")
            return []

        formatted_results = []
        for i, res in enumerate(results):
            formatted_result = {
                "result_number": i + 1,
                "source": res["metadata"].get("source", "N/A"),
                "chunk_index": res["metadata"].get("chunk_index", "N/A"),
                "timestamp": res["metadata"].get("timestamp", "N/A"),
                "distance": f"{res['distance']:.4f}",
                "document": res["document"],
            }
            if "tags" in res["metadata"]:
                formatted_result["tags"] = res["metadata"]["tags"]
            formatted_results.append(formatted_result)
        return formatted_results
    except Exception as e:
        logger.error(f"Error during query: {e}")
        return [{"error": f"Failed to perform query: {e}"}]


@mcp.tool()
async def reset_vector_store() -> str:
    """Resets (clears) the entire vector store.

    This action is irreversible and will delete all ingested documents from the RAG system.

    Returns:
        A confirmation message that the vector store has been reset.
    """
    try:
        await rag_system.reset_vector_store()
        return "Vector store has been successfully reset."
    except Exception as e:
        logger.error(f"Error resetting vector store: {e}")
        return f"Failed to reset vector store: {e}"
