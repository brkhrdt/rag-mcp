import argparse
from pathlib import Path
from src.rag_mcp.rag_system import RAGSystem

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

    # Ingest command
    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest a document into the RAG system"
    )
    ingest_parser.add_argument(
        "file_path", type=str, help="Path to the document file to ingest"
    )
    ingest_parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Maximum token size for each text chunk (default: 512)",
    )
    ingest_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Token overlap between chunks (default: 50)",
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

    rag_system = RAGSystem(chroma_persist_directory=args.db_path)

    if args.command == "ingest":
        file_path = Path(args.file_path)
        if not file_path.exists():
            print(f"Error: File not found at {file_path}")
            return
        rag_system.ingest(file_path, args.chunk_size, args.chunk_overlap)
    elif args.command == "query":
        results = rag_system.query(args.query_text, args.num_results)
        if results:
            print("\n--- Query Results ---")
            for i, res in enumerate(results):
                print(f"\nResult {i+1}:")
                print(f"  Source: {res['metadata'].get('source', 'N/A')}")
                print(f"  Chunk Index: {res['metadata'].get('chunk_index', 'N/A')}")
                print(f"  Distance: {res['distance']:.4f}")
                print(f"  Document: {res['document']}")
        else:
            print("No results found for your query.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
