# Retrieval Augmented Generation (RAG) Project

## Model Context Protocol (MCP)
### API
*   **`ingest(file)`**: Ingests file content into the database.
*   **`query(string, num_matches)`**: Searches database, returns `num_matches` relevant results.

## RAG Implementation
*   Built with `transformers` Python library.
*   Data stored in Parquet files and embeddings are searched to find nearest to embedding of query text
*   **Text Chunking**:
    *   Splits text into chunks fitting embedding model's max tokens.
    *   Uses a default embedding model (TBD).
    *   Configurable overlap (e.g., 30%) between chunks to maintain context.

---

## Project Embellishment: Architectural Vision and Solution Approach

The core idea of this project is to combine a powerful language model with a retrieval mechanism to provide more accurate and contextually relevant answers. The initial `README.md` provides a good foundation with `ingest` and `query` functions, and mentions `transformers`, Parquet, and text chunking.

### Architectural Vision

We will build a modular RAG system with clear separation of concerns:

1.  **Data Ingestion & Preprocessing (`ingest`):**
    *   Handles various document types (PDF, TXT, DOCX, etc.).
    *   Robust text extraction.
    *   Intelligent text chunking.
    *   Embedding generation.
    *   Storage of chunks and embeddings.
2.  **Retrieval (`query`):
    *   Embeds the query.
    *   Efficiently searches for similar embeddings.
    *   Retrieves relevant text chunks.
3.  **Generation:**
    *   Feeds retrieved chunks and the original query to a Large Language Model (LLM).
    *   Generates a coherent and informed response.
4.  **Configuration & Models:**
    *   Flexible configuration for embedding models, LLMs, chunking parameters.
    *   Ability to swap models easily.

### Detailed Solution Approach

Let's break down the implementation into key components and how they interact.

#### 1. Core Components & Their Responsibilities

*   **`DocumentProcessor`**:
    *   **Responsibility**: Extracting raw text from various file types.
    *   **Implementation**: Use libraries like `pypdf`, `python-docx`, etc.
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
    *   **Responsibility**: Orchestrating the entire RAG workflow: ingestion, retrieval, and generation.
    *   **Implementation**:
        *   Expose `ingest(file_path)`:
            1.  `DocumentProcessor` extracts text.
            2.  `TextChunker` chunks text.
            3.  `EmbeddingModel` embeds chunks.
            4.  `VectorStore` stores chunks and embeddings.
        *   Expose `query(string, num_matches)`:
            1.  `EmbeddingModel` embeds query.
            2.  `VectorStore` retrieves top `num_matches` relevant chunks.
            3.  `LLMGenerator` generates response using query and retrieved chunks.
*   **`LLMGenerator`**:
    *   **Responsibility**: Taking the user query and retrieved context, and generating a coherent answer using an LLM.
    *   **Implementation**:
        *   Wrap a `transformers` causal language model (e.g., `Llama-2`, `Mistral`, `GPT-2` for local testing).
        *   Construct a prompt that includes the query and the retrieved context.
        *   Handle token limits for the LLM.

#### 2. Key Libraries & Tools

*   **`transformers`**: For loading and using pre-trained models (embedding and LLM).
*   **`sentence-transformers`**: (Optional, but highly recommended) Simplifies using transformer models for embeddings.
*   **`pypdf`, `python-docx`, `openpyxl`**: For document parsing.
*   **`chromadb`**: For the vector store (local, easy to set up).
*   **`tiktoken` / `transformers` tokenizers**: For accurate token counting for chunking.
*   **`pathlib`**: For robust file path handling.

#### 3. Project Structure (Refined)

#### 4. Class and Function Descriptions

*   **`DocumentProcessor` Class**:
    *   **Purpose**: Responsible for extracting raw text content from various document formats (e.g., PDF, DOCX).
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

*   **`LLMGenerator` Class**:
    *   **Purpose**: Utilizes a Large Language Model (LLM) to generate coherent and informed responses based on a user query and retrieved contextual information.
    *   **Key Functions**:
        *   `generate_response(query: str, context: List[str]) -> str`: Takes the user's `query` and a list of `context` strings (retrieved chunks) and generates a natural language response.

*   **`RAGSystem` Class (or `main.py` orchestration)**:
    *   **Purpose**: Orchestrates the entire Retrieval Augmented Generation workflow, integrating all other components from ingestion to query response.
    *   **Key Functions**:
        *   `ingest(file_path: str)`: Processes a document from the given `file_path`, extracts text, chunks it, generates embeddings, and stores them in the `VectorStore`.
        *   `query(user_query: str, num_matches: int = 5) -> str`: Takes a `user_query`, retrieves the `num_matches` most relevant chunks from the `VectorStore`, and uses the `LLMGenerator` to produce a final answer.
