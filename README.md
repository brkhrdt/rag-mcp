# Retrieval Augmented Generation (RAG) Project

This project implements a Retrieval Augmented Generation (RAG) system, accessible via a Model Context Protocol (MCP).

## Model Context Protocol (MCP)

The MCP provides the following functions:

*   **`ingest(file)`**: Ingests the content of a given file into the project's database.
*   **`query(string)`**: Searches the database for the most relevant information based on the input string and returns the top 5 matches.

## RAG Implementation Details

The RAG system is built using the `transformers` Python library. Data is stored and managed using `pandas`, specifically saving embeddings mapped to text chunks in a Parquet file.

Text chunking for embedding is performed by splitting the input text into chunks that fit within the maximum token limit of the chosen embedding model. A default embedding model (to be determined) will be used. To maintain context across chunks, a configurable overlap is applied. For example, a 30% overlap means the tail 30% of the previous chunk will be the leading 30% of the current chunk. This overlapping strategy helps ensure that important information is not lost at chunk boundaries.
