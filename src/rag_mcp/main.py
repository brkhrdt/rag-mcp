import argparse
from pathlib import Path
from rag_mcp.rag import RAG


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
        "file_path", type=str, help="Path to the document file to ingest"
    )
    ingest_file_parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Maximum token size for each text chunk (default: 512)",
    )
    ingest_file_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Token overlap between chunks (default: 50)",
    )
    ingest_file_parser.add_argument(  # Add tags argument for ingest-file
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
        default=512,
        help="Maximum token size for each text chunk (default: 512)",
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
    ingest_text_parser.add_argument(  # Add tags argument for ingest-text
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
        file_path = Path(args.file_path)
        rag_system.ingest_file(
            file_path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            tags=args.tags,  # Pass tags
        )
    elif args.command == "ingest-text":
        rag_system.ingest_string(
            args.text_content,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            source_name=args.source_name,
            tags=args.tags,  # Pass tags
        )
    elif args.command == "query":
        results = rag_system.query(args.query_text, args.num_results)
        if results:
            print("\n--- Query Results ---")
            for i, res in enumerate(results):
                print(f"\nResult {i + 1}:")
                print(f"  Source: {res['metadata'].get('source', 'N/A')}")
                print(f"  Chunk Index: {res['metadata'].get('chunk_index', 'N/A')}")
                print(
                    f"  Timestamp: {res['metadata'].get('timestamp', 'N/A')}"
                )  # Print timestamp
                if "tags" in res["metadata"]:  # Print tags if present
                    print(f"  Tags: {', '.join(res['metadata']['tags'])}")
                print(f"  Distance: {res['distance']:.4f}")
                print(f"  Document: {res['document']}")
        else:
            print("No results found for your query.")


if __name__ == "__main__":
    main()
