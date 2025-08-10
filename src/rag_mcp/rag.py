from pathlib import Path
from typing import Any, Dict, List, Optional
import datetime
import logging

from rag_mcp.document_processor import DocumentProcessor
from rag_mcp.embedding_model import EmbeddingModel
from rag_mcp.text_chunker import TextChunker
from rag_mcp.vector_store import VectorStore

# Get a logger for this module
logger = logging.getLogger(__name__)

class RAG:
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
        self.default_chunk_size = self.embedding_model.max_input_tokens

    def _process_and_ingest(
        self,
        text: str,
        source_name: str,
        chunk_size: int,
        chunk_overlap: int,
        tags: Optional[List[str]] = None,
    ):
        """Helper method to process text, chunk, embed, and add to vector store."""
        if not text.strip():
            logger.warning(
                f"Provided text content for '{source_name}' is empty. Skipping ingestion."
            )
            return

        model_max_tokens = self.embedding_model.max_input_tokens
        effective_chunk_size = min(chunk_size, model_max_tokens)

        if effective_chunk_size < chunk_size:
            logger.warning(
                f"Requested chunk_size ({chunk_size}) exceeds embedding model's max input tokens ({model_max_tokens}). "
                f"Using effective_chunk_size of {effective_chunk_size}."
            )

        chunks = self.text_chunker.chunk_text(
            text, chunk_size=effective_chunk_size, chunk_overlap=chunk_overlap
        )

        if not chunks:
            logger.warning(f"No chunks generated from {source_name}. Skipping ingestion.")
            return

        embeddings = self.embedding_model.embed(chunks)

        current_time = datetime.datetime.now().isoformat()
        metadatas = []
        for i in range(len(chunks)):
            metadata = {
                "source": source_name,
                "chunk_index": i,
                "timestamp": current_time,
            }
            if tags:
                metadata["tags"] = ",".join(tags)
            metadatas.append(metadata)

        self.vector_store.add_documents(chunks, embeddings, metadatas)
        logger.info(f"Ingested {len(chunks)} chunks from {source_name}")

    def ingest_string(
        self,
        text_content: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: int = 50,
        source_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        """Ingest a text string into the RAG system."""
        ingested_source_name = source_name if source_name else "string_input"

        actual_chunk_size = (
            chunk_size if chunk_size is not None else self.default_chunk_size
        )
        self._process_and_ingest(
            text_content,
            ingested_source_name,
            actual_chunk_size,
            chunk_overlap,
            tags,
        )

    def ingest_file(
        self,
        file_path: Path,
        chunk_size: Optional[int] = None,
        chunk_overlap: int = 50,
        tags: Optional[List[str]] = None,
    ):
        """Ingest a document file into the RAG system."""
        if not file_path.exists():
            logger.error(f"Error: File not found at {file_path}")
            return
        try:
            text = self.document_processor.extract_text(file_path)
            ingested_source_name = str(file_path)

            actual_chunk_size = (
                chunk_size if chunk_size is not None else self.default_chunk_size
            )
            self._process_and_ingest(
                text,
                ingested_source_name,
                actual_chunk_size,
                chunk_overlap,
                tags,
            )
        except ValueError as e:
            logger.error(f"Error ingesting file {file_path}: {e}")
            return

    def query(self, query_text: str, num_results: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.embed(query_text)
        results = self.vector_store.query(query_embedding, num_results)
        return results

    def reset_vector_store(self):
        self.vector_store.reset()
        logger.info("Vector store has been reset.")
