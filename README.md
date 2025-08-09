# Retrieval Augmented Generation (RAG) Project

## Model Context Protocol (MCP)
### API
*   **`ingest(file)`**: Ingests file content into the database.
*   **`query(string)`**: Searches database, returns top 5 relevant matches.

## RAG Implementation
*   Built with `transformers` Python library.
*   Data stored in Parquet files and embeddings are searched to find nearest to embedding of query text
*   **Text Chunking**:
    *   Splits text into chunks fitting embedding model's max tokens.
    *   Uses a default embedding model (TBD).
    *   Configurable overlap (e.g., 30%) between chunks to maintain context.

