# src/rag-mcp/rag_system.py
from pathlib import Path
from typing import List, Dict, Any

from .document_processor import DocumentProcessor
from .text_chunker import TextChunker
from .embedding_model import EmbeddingModel
from .vector_store import VectorStore

class RAGSystem:
    """
    Orchestrates the entire RAG workflow: ingestion and retrieval.
    """

    def __init__(self,
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 chroma_collection_name: str = "rag_collection",
                 chroma_persist_directory: str = "chroma_db"):
        """
        Initializes the RAGSystem with its core components.

        Args:
            embedding_model_name (str): The name of the sentence-transformers model for embeddings.
            chroma_collection_name (str): The name of the ChromaDB collection.
            chroma_persist_directory (str): The directory for ChromaDB persistence.
        """
        self.document_processor = DocumentProcessor()
        self.text_chunker = TextChunker()
        self.embedding_model = EmbeddingModel(model_name=embedding_model_name)
        self.vector_store = VectorStore(
            collection_name=chroma_collection_name,
            persist_directory=chroma_persist_directory
        )

    def ingest(self, file_path: Path, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Ingests a document into the RAG system.

        Args:
            file_path (Path): The path to the document to ingest.
            chunk_size (int): The maximum token size for each text chunk.
            chunk_overlap (int): The token overlap between chunks.
        """
        print(f"Ingesting document: {file_path}")
        try:
            # 1. Extract text
            raw_text = self.document_processor.extract_text(file_path)

            # 2. Chunk text
            chunks = self.text_chunker.chunk_text(raw_text, chunk_size, chunk_overlap)
            print(f"Split document into {len(chunks)} chunks.")

            # 3. Embed chunks
            embeddings = self.embedding_model.embed(chunks)
            print(f"Generated embeddings for {len(embeddings)} chunks.")

            # 4. Store chunks and embeddings
            metadatas = [{"source": str(file_path), "chunk_index": i} for i in range(len(chunks))]
            ids = [f"{file_path.stem}_chunk_{i}" for i in range(len(chunks))]
            self.vector_store.add_documents(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)
            print(f"Successfully ingested {file_path}")

        except Exception as e:
            print(f"Error ingesting {file_path}: {e}")

    def query(self, query_text: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Queries the RAG system for relevant information.

        Args:
            query_text (str): The query string.
            num_results (int): The number of top relevant results to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing 'document', 'metadata', and 'distance'.
        """
        print(f"Processing query: '{query_text}'")
        # 1. Embed query
        query_embedding = self.embedding_model.embed(query_text)

        # 2. Retrieve relevant chunks
        results = self.vector_store.query(query_embedding, num_results)
        print(f"Retrieved {len(results)} relevant chunks.")
        return results

    def reset_vector_store(self):
        """Resets the underlying vector store."""
        self.vector_store.reset()

