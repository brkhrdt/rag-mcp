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
    *   **Responsibility**: Converting text chunks and queries into numerical vector embeddings.
    *   **Implementation**:
        *   Wrap a `transformers` model (e.g., `SentenceTransformer` from `sentence-transformers` library for ease of use, or directly `AutoModel` and `AutoTokenizer`).
        *   Provide methods for batch embedding.
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

#### 3. Project Structure (Refined)

#### 4. Class and Function Descriptions

*   **`DocumentProcessor` Class**:
    *   **Purpose**: Responsible for extracting raw text content from various document formats. Initially, this will support plain text files.
    *   **Key Functions**:
        *   `extract_text(file_path: str) -> str`: Takes a file path and returns the extracted plain text content.

*   **`TextChunker` Class**:
    *   **Purpose**: Splits long text documents into smaller, manageable chunks suitable for embedding models, ensuring proper token limits and configurable overlap.
    *   **Key Functions**:
        *   `chunk_text(text: str, max_tokens: int, overlap: float) -> List[str]`: Divides the input text into a list of text chunks, respecting `max_tokens` and applying `overlap`.

*   **`EmbeddingModel` Class**:
    *   **Purpose**: Converts text (chunks or queries) into numerical vector embeddings using a pre-trained transformer model.
    *   **Key Functions**:
        *   `encode(texts: Union[str, List[str]]) -> np.ndarray`: Takes a single string or a list of strings and returns their corresponding embeddings as a NumPy array.

*   **`VectorStore` Class**:
    *   **Purpose**: Stores text chunks and their associated embeddings, and facilitates efficient similarity searches to retrieve relevant chunks.
    *   **Key Functions**:
        *   `add_documents(documents: List[str], embeddings: List[np.ndarray], metadatas: Optional[List[Dict]] = None)`: Adds text documents, their embeddings, and optional metadata to the store.
        *   `search(query_embedding: np.ndarray, num_results: int) -> List[Dict]`: Searches the store for the `num_results` most similar documents to the `query_embedding`, returning the relevant chunks and their metadata.

*   **`RAGSystem` Class (or `main.py` orchestration)**:
    *   **Purpose**: Orchestrates the entire Retrieval Augmented Generation workflow, integrating all other components from ingestion to query response.
    *   **Key Functions**:
        *   `ingest(file_path: str)`: Processes a document from the given `file_path`, extracts text, chunks it, generates embeddings, and stores them in the `VectorStore`.
        *   `query(user_query: str, num_matches: int = 5) -> List[Dict]`: Takes a `user_query`, retrieves the `num_matches` most relevant chunks from the `VectorStore`, and returns them.
