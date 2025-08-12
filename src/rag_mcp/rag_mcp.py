from typing import Dict, Optional, List
from mcp.server.fastmcp import FastMCP
import logging

from rag_mcp.rag import RAG

# Get a logger for this module
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("interactive-shell")

# Initialize RAG system as None initially, to be set up later or mocked
# This allows tests to replace it before any tool functions are called
rag_system: Optional[RAG] = None


def _get_rag_system() -> RAG:
    """Lazily initializes and returns the RAG system instance."""
    global rag_system
    if rag_system is None:
        # TODO: Make db_path configurable for MCP functions if needed
        rag_system = RAG()
    return rag_system


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
    current_rag_system = _get_rag_system()
    ingested_files, skipped_files = current_rag_system.ingest_files(
        file_paths,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tags=tags,
    )

    response = f"Ingested {len(ingested_files)} file(s)."
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
    current_rag_system = _get_rag_system()
    try:
        await current_rag_system.ingest_string(
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
    current_rag_system = _get_rag_system()
    try:
        results = current_rag_system.query(query_text, num_results)
        if not results:
            logger.info("No results found for your query.")
            return []

        return [result.to_dict() for result in results]
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
    current_rag_system = _get_rag_system()
    try:
        await current_rag_system.reset_vector_store()
        return "Vector store has been successfully reset."
    except Exception as e:
        logger.error(f"Error resetting vector store: {e}")
        return f"Failed to reset vector store: {e}"
