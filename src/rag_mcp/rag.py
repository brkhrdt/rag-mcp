from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, cast, Mapping
import datetime
import logging
import glob

from rag_mcp.document_processor import DocumentProcessor
from rag_mcp.embedding_model import EmbeddingModel
from rag_mcp.text_chunker import TextChunker
from rag_mcp.vector_store import VectorStore

# Get a logger for this module
logger = logging.getLogger(__name__)


class QueryResult:
    def __init__(
        self,
        document: str,
        metadata: Dict[str, Any],
        distance: float,
        result_number: int = 1,
    ):
        self.document = document
        self.metadata = metadata
        self.distance = distance
        self.result_number = result_number

    def __str__(self) -> str:
        """Format result for CLI display."""
        lines = [
            f"Result {self.result_number}:",
            f"Source: {self.metadata.get('source', 'N/A')}",
            f"Chunk Index: {self.metadata.get('chunk_index', 'N/A')}",
            f"Timestamp: {self.metadata.get('timestamp', 'N/A')}",
        ]

        if "tags" in self.metadata:
            tags = self.metadata["tags"]
            if isinstance(tags, str):
                tags = tags.split(",")
            lines.append(f"Tags: {', '.join(tags)}")

        lines.extend([f"Distance: {self.distance:.4f}", f"Document:\n{self.document}"])

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Format result for MCP response."""
        result = {
            "result_number": self.result_number,
            "document": self.document,
            "distance": f"{self.distance:.4f}",
            "metadata": self.metadata,  # Return the entire metadata dictionary
        }

        # Handle tags within the metadata dictionary if they exist and are a string
        if "tags" in result["metadata"] and isinstance(result["metadata"]["tags"], str):
            result["metadata"]["tags"] = result["metadata"]["tags"].split(",")

        return result


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
            logger.warning(
                f"No chunks generated from {source_name}. Skipping ingestion."
            )
            return

        embeddings: List[Sequence[float]] = cast(
            List[Sequence[float]], self.embedding_model.embed(chunks)
        )

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

        self.vector_store.add_documents(
            chunks, embeddings, cast(List[Mapping], metadatas)
        )
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

    def ingest_files(
        self,
        file_patterns: List[str],
        chunk_size: Optional[int] = None,
        chunk_overlap: int = 50,
        tags: Optional[List[str]] = None,
    ) -> tuple[List[str], List[str]]:
        """Ingest multiple files using glob patterns.

        Returns:
            Tuple of (ingested_files, skipped_files)
        """
        ingested_files = []
        skipped_files = []

        for pattern in file_patterns:
            for file_path_str in glob.glob(pattern):
                file_path = Path(file_path_str)
                if file_path.is_file():
                    try:
                        self.ingest_file(
                            file_path,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            tags=tags,
                        )
                        ingested_files.append(str(file_path))
                    except Exception as e:
                        logger.error(f"Error ingesting {file_path}: {e}")
                        skipped_files.append(f"{file_path} (Error: {e})")
                else:
                    logger.warning(f"Skipping non-file path: {file_path}")
                    skipped_files.append(f"{file_path} (Skipped: Not a file)")

        return ingested_files, skipped_files

    def query(self, query_text: str, num_results: int = 5) -> List[QueryResult]:
        query_embedding: Sequence[float] = cast(
            Sequence[float],
            self.embedding_model.embed(query_text, show_progress_bar=False),
        )
        results = self.vector_store.query(query_embedding, num_results)

        query_results = []
        for i, res in enumerate(results):
            query_results.append(
                QueryResult(
                    document=res["document"],
                    metadata=res["metadata"],
                    distance=res["distance"],
                    result_number=i + 1,
                )
            )

        return query_results

    def reset_vector_store(self):
        self.vector_store.reset()
        logger.info("Vector store has been reset.")
