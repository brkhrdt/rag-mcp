# MCPRAG

This project implements a Retrieval Augmented Generation (RAG) system focused on the retrieval component. It provides functionality to ingest various document types, process them into searchable chunks, and retrieve the most relevant information based on a given query. The system is designed with modularity, allowing for flexible integration with different embedding models and vector stores.

## Project Embellishment: Architectural Vision and Solution Approach

The core idea of this project is to combine a powerful language model with a retrieval mechanism to provide more accurate and contextually relevant answers. The initial `README.md` provides a good foundation with `ingest` and `query` functions, and mentions `transformers`, Parquet, and text chunking.

### Architectural Vision

We will build a modular RAG system with clear separation of concerns:

1.  **Data Ingestion & Preprocessing (`ingest`):**
    *   Handles various document types (currently plain text, with PDF, DOCX, etc., as future improvements).
    *   Robust text extraction.
    *   Intelligent text chunking.
    *   Embedding generation.
    *   Storage of chunks and embeddings.
2.  **Retrieval (`query`):
    *   Embeds the query.
    *   Efficiently searches for similar embeddings.
    *   Retrieves relevant text chunks.
3.  **Configuration & Models:**
    *   Flexible configuration for embedding models, LLMs, chunking parameters.
    *   Ability to swap models easily.

### Detailed Solution Approach

Let's break down the implementation into key components and how they interact.

#### 1. Core Components & Their Responsibilities

*   **`DocumentProcessor`**:
    *   **Responsibility**: Extracting raw text from various file types.
    *   **Implementation**: Initially, this will focus on plain text files. Support for other formats like `pypdf`, `python-docx`, etc., will be future enhancements.
*   **`TextChunker`**:
    *   **Responsibility**: Splitting raw text into manageable, overlapping chunks suitable for embedding models.
    *   **Implementation**:
        *   Utilize `transformers` tokenizers to count tokens accurately.
        *   Implement a sliding window approach for overlap.
        *   Consider strategies to preserve semantic boundaries (e.g., splitting by paragraphs/sentences first).
*   **`EmbeddingModel`**:
    *   **Responsibility**: Converting text chunks and queries into numerical vector embeddings, with support for GPU acceleration.
    *   **Implementation**:
        *   Wrap a `transformers` model (e.g., `SentenceTransformer` from `sentence-transformers` library for ease of use, or directly `AutoModel` and `AutoTokenizer`).
        *   Provide methods for batch embedding.
        *   Ensure the model can leverage available GPUs for faster processing.
*   **`VectorStore`**:
    *   **Responsibility**: Storing text chunks and their corresponding embeddings, and performing efficient similarity searches.
    *   **Implementation**:
        *   While Parquet files were initially mentioned, a dedicated vector database is better for scalability and performance. We will aim for `ChromaDB` for its ease of use and local persistence.
*   **`RAGSystem` (Or `main.py` orchestrator):**
    *   **Responsibility**: Orchestrating the entire RAG workflow: ingestion and retrieval.
    *   **Implementation**:
        *   Expose `ingest(file_path)`:
            1.  `DocumentProcessor` extracts text.
            2.  `TextChunker` chunks text.
            3.  `EmbeddingModel` embeds chunks.
            4.  `VectorStore` stores chunks and embeddings.
        *   Expose `query(string, num_matches)`:
            1.  `EmbeddingModel` embeds query.
            2.  `VectorStore` retrieves top `num_matches` relevant chunks.

#### 2. Key Libraries & Tools

*   **`transformers`**: For loading and using pre-trained models (embedding and LLM).
*   **`sentence-transformers`**: (Optional, but highly recommended) Simplifies using transformer models for embeddings.
*   **`pypdf`, `python-docx`, `openpyxl`**: (Future enhancements) For document parsing.
*   **`chromadb`**: For the vector store (local, easy to set up).
*   **`tiktoken` / `transformers` tokenizers**: For accurate token counting for chunking.
*   **`pathlib`**: For robust file path handling.
*   **`torch` / `tensorflow`**: (Implicitly, via `transformers`) For GPU acceleration.

#### 3. Project Structure

-   `.gitignore`
-   `main.py`
-   `pyproject.toml`
-   `README.md`
-   `src/`
    -   `rag-mcp/`
        -   `__init__.py`
        -   `document_processor.py`
        -   `text_chunker.py`
        -   `embedding_model.py`
        -   `vector_store.py`
        -   `rag_system.py`
-   `tests/`
    -   `__init__.py`
    -   `test_document_processor.py`
    -   `test_text_chunker.py`
    -   `test_embedding_model.py`
    -   `test_vector_store.py`
    -   `test_rag_system.py`

## Usage

To use the RAG system, you'll typically follow these steps:

1.  **Initialize the RAGSystem:**
    ```python
    from src.rag_mcp.rag_system import RAGSystem
    rag_system = RAGSystem()
    ```

2.  **Ingest Documents:**
    Prepare your documents (e.g., `.txt` files) and ingest them.
    ```python
    from pathlib import Path
    # Create a dummy text file for demonstration
    with open("example.txt", "w") as f:
        f.write("This is a sample document about the capital of France. Paris is the capital and largest city of France.")

    file_path = Path("example.txt")
    rag_system.ingest(file_path)
    ```
    You can ingest multiple documents. Each document will be processed, chunked, embedded, and stored in the vector store.

3.  **Query the System:**
    Once documents are ingested, you can query the system to retrieve relevant information.
    ```python
    query_text = "What is the capital of France?"
    results = rag_system.query(query_text, num_results=2)

    print("\nQuery Results:")
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Document: {result['document']}")
        print(f"  Metadata: {result['metadata']}")
        print(f"  Distance: {result['distance']:.4f}")
    ```

4.  **Resetting the Vector Store (Optional):**
    If you want to clear all ingested data and start fresh, you can reset the vector store.
    ```python
    rag_system.reset_vector_store()
    print("\nVector store has been reset.")
    ```

**Note:** Ensure you have all necessary dependencies installed (`pip install -r requirements.txt` or similar, once a `requirements.txt` is defined). The `chroma_db` directory will be created in your project root to persist the vector store data.
