import argparse
from pathlib import Path
import glob
import logging
import sys  # Import sys to handle SystemExit for testing

from rag_mcp.rag import RAG

# Get a logger for this module
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="RAG System CLI for document ingestion and querying."
    )

    parser.add_argument(
        "--db-path",
        type=str,
        default="chroma_db",
        help="Path to the ChromaDB persistence directory (default: chroma_db)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest File command
    ingest_file_parser = subparsers.add_parser(
        "ingest-file", help="Ingest a document file into the RAG system"
    )
    ingest_file_parser.add_argument(
        "file_paths",
        type=str,
        nargs="+",
        help="Path(s) to the document file(s) to ingest (supports glob patterns)",
    )
    ingest_file_parser.add_argument(
        "--chunk-size",
        type=int,
        help="Maximum token size for each text chunk",
    )
    ingest_file_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Token overlap between chunks (default: 50)",
    )
    ingest_file_parser.add_argument(
        "--tags",
        type=str,
        nargs="*",
        help="Optional space-separated tags to associate with the ingested file (e.g., --tags 'report' 'Q1')",
    )

    # Ingest Text command
    ingest_text_parser = subparsers.add_parser(
        "ingest-text", help="Ingest a text string directly into the RAG system"
    )
    ingest_text_parser.add_argument(
        "text_content", type=str, help="The text string to ingest"
    )
    ingest_text_parser.add_argument(
        "--chunk-size",
        type=int,
        help="Maximum token size for each text chunk",
    )
    ingest_text_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Token overlap between chunks (default: 50)",
    )
    ingest_text_parser.add_argument(
        "--source-name",
        type=str,
        default=None,
        help="Optional name for the source when ingesting a string (default: 'string_input')",
    )
    ingest_text_parser.add_argument(
        "--tags",
        type=str,
        nargs="*",
        help="Optional space-separated tags to associate with the ingested text (e.g., --tags 'email' 'urgent')",
    )

    # Query command
    query_parser = subparsers.add_parser(
        "query", help="Query the RAG system for relevant information"
    )
    query_parser.add_argument("query_text", type=str, help="The query string")
    query_parser.add_argument(
        "--num-results",
        type=int,
        default=5,
        help="Number of top relevant results to retrieve (default: 5)",
    )

    args = parser.parse_args()

    rag_system = RAG(chroma_persist_directory=args.db_path)

    if args.command == "ingest-file":
        ingested_files, skipped_files = rag_system.ingest_files(
            args.file_paths,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            tags=args.tags,
        )
        # Log results if needed, but no output for CLI consistency
    elif args.command == "ingest-text":
        rag_system.ingest_string(
            args.text_content,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            source_name=args.source_name,
            tags=args.tags,
        )
    elif args.command == "query":
        results = rag_system.query(args.query_text, args.num_results)
        if results:
            logger.info("\n--- Query Results ---")
            for result in results:
                print(f"\n{result}")
        else:
            print("No results found for your query.")
    else:
        # If no command is provided, print help and exit with an error code
        parser.print_help(sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
