# main.py
from pathlib import Path
from src.rag_mcp.rag_system import RAGSystem


def main():
    print("Starting MCPRAG system...")

    # Initialize the RAG system
    rag_system = RAGSystem()

    # --- Example Usage ---

    # 1. Create a dummy text file for ingestion
    dummy_file_path = Path("example_document.txt")
    with open(dummy_file_path, "w", encoding="utf-8") as f:
        f.write(
            "This is the first sentence about artificial intelligence. "
            "AI is transforming various industries. "
            "Machine learning is a subset of AI. "
            "Deep learning is a more advanced form of machine learning. "
            "Natural Language Processing (NLP) is a key area in AI. "
            "Robotics also heavily utilizes AI principles. "
            "The future of technology is intertwined with AI development."
        )
    print(f"Created dummy document: {dummy_file_path}")

    # 2. Ingest the document
    rag_system.ingest(dummy_file_path)

    # 3. Perform a query
    query_text = "What is machine learning?"
    results = rag_system.query(query_text, num_results=2)

    print("\n--- Query Results ---")
    if results:
        for i, result in enumerate(results):
            print(f"Result {i + 1} (Distance: {result['distance']:.4f}):")
            print(
                f"  Source: {result['metadata'].get('source', 'N/A')}, Chunk Index: {result['metadata'].get('chunk_index', 'N/A')}"
            )
            print(f"  Document: {result['document']}\n")
    else:
        print("No results found.")

    # Clean up the dummy file
    dummy_file_path.unlink(missing_ok=True)
    print(f"Cleaned up dummy document: {dummy_file_path}")

    # Optional: Reset the vector store if you want to clear all data
    # rag_system.reset_vector_store()
    # print("Vector store has been reset.")

    print("MCPRAG system finished.")


if __name__ == "__main__":
    main()
