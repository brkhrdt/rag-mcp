from pathlib import Path
from typing import Any, Dict, List

from src.rag_mcp.document_processor import DocumentProcessor
from src.rag_mcp.embedding_model import EmbeddingModel
from src.rag_mcp.text_chunker import TextChunker
from src.rag_mcp.vector_store import VectorStore


class RAG:
    """
    Orchestrates the entire RAG workflow: ingestion and retrieval.
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        chroma_collection_name: str = "rag_collection",
        chroma_persist_directory: str = "chroma_db",
    ):
        self.document_processor = DocumentProcessor()
        self.embedding_model = EmbeddingModel(embedding_model_name)
        self.text_chunker = TextChunker()
        self.vector_store = VectorStore(
            chroma_collection_name, chroma_persist_directory
        )

    def ingest(self, file_path: Path, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Ingests a document by extracting text, chunking it, embedding chunks,
        and adding them to the vector store.

        Args:
            file_path: The path to the document file.
            chunk_size: The desired size of text chunks in tokens.
            chunk_overlap: The number of tokens to overlap between consecutive chunks.
        """
        if not file_path.exists():
            print(f"Error: File not found at {file_path}")
            return

        try:
            text = self.document_processor.extract_text(file_path)
        except ValueError as e:
            print(f"Error ingesting file {file_path}: {e}")
            return

        # Ensure chunk_size does not exceed the embedding model's max input tokens
        model_max_tokens = self.embedding_model.max_input_tokens
        effective_chunk_size = min(chunk_size, model_max_tokens)

        if effective_chunk_size < chunk_size:
            print(
                f"Warning: Requested chunk_size ({chunk_size}) exceeds embedding model's max input tokens ({model_max_tokens}). "
                f"Using effective_chunk_size of {effective_chunk_size}."
            )

        chunks = self.text_chunker.chunk_text(
            text, chunk_size=effective_chunk_size, chunk_overlap=chunk_overlap
        )

        if not chunks:
            print(f"No chunks generated from {file_path}. Skipping ingestion.")
            return

        # Generate embeddings for the chunks
        embeddings = self.embedding_model.embed(chunks)

        # Prepare metadata (e.g., source file)
        metadatas = [
            {"source": str(file_path), "chunk_index": i} for i in range(len(chunks))
        ]

        # Add to vector store
        self.vector_store.add_documents(chunks, embeddings, metadatas)
        print(f"Ingested {len(chunks)} chunks from {file_path}")

    def query(self, query_text: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Queries the RAG system with a given text, retrieves relevant documents,
        and returns them.

        Args:
            query_text: The user's query string.
            num_results: The number of top relevant results to retrieve.

        Returns:
            A list of dictionaries, where each dictionary contains the content
            and metadata of a retrieved chunk.
        """
        query_embedding = self.embedding_model.embed(query_text)
        results = self.vector_store.query(query_embedding, num_results)
        return results

    def reset_vector_store(self):
        """
        Resets (clears) the entire vector store.
        """
        self.vector_store.reset()
        print("Vector store has been reset.")
