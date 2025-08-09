# src/rag-mcp/vector_store.py
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional


class VectorStore:
    """
    Stores text chunks and their corresponding embeddings using ChromaDB,
    and performs efficient similarity searches.
    """

    def __init__(
        self,
        collection_name: str = "rag_collection",
        persist_directory: str = "chroma_db",
    ):
        """
        Initializes the VectorStore with a ChromaDB client.

        Args:
            collection_name (str): The name of the collection to use.
            persist_directory (str): The directory where ChromaDB will store its data.
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        # Note: ChromaDB's default embedding function is not used here,
        # as we will provide pre-computed embeddings from our EmbeddingModel.
        # We pass None for embedding_function to indicate we'll handle embeddings.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            ),  # This is a placeholder, we will override it with our own embeddings
        )
        # When adding documents, we will explicitly pass embeddings.
        # The embedding_function in get_or_create_collection is primarily for when you let ChromaDB embed for you.
        # For querying, if you pass query_embeddings, it uses those. If not, it uses the collection's embedding_function.
        # So, it's safer to initialize with a matching embedding function or ensure we always pass query_embeddings.
        # For now, we'll keep it simple and assume we always pass embeddings.

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ):
        """
        Adds documents (chunks), their embeddings, and optional metadata to the vector store.

        Args:
            documents (List[str]): The text content of the chunks.
            embeddings (List[List[float]]): The corresponding embeddings for each chunk.
            metadatas (Optional[List[Dict[str, Any]]]): Optional list of metadata dictionaries for each chunk.
            ids (Optional[List[str]]): Optional list of unique IDs for each chunk. If None, ChromaDB generates them.
        """
        if not ids:
            # Generate simple IDs if not provided
            ids = [f"doc_{i}" for i in range(len(documents))]

        if (
            len(documents) != len(embeddings)
            or (metadatas and len(documents) != len(metadatas))
            or len(documents) != len(ids)
        ):
            raise ValueError(
                "Lengths of documents, embeddings, metadatas, and ids must match."
            )

        self.collection.add(
            documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids
        )
        print(
            f"Added {len(documents)} documents to ChromaDB collection '{self.collection.name}'."
        )

    def query(
        self, query_embedding: List[float], num_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Performs a similarity search against the stored embeddings.

        Args:
            query_embedding (List[float]): The embedding of the query.
            num_results (int): The number of top relevant results to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing 'document', 'metadata', and 'distance'.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=num_results,
            include=["documents", "metadatas", "distances"],
        )

        # Format results for easier consumption
        formatted_results = []
        if results and results["documents"]:
            for i in range(len(results["documents"][0])):
                formatted_results.append(
                    {
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                    }
                )
        return formatted_results

    def reset(self):
        """
        Resets (deletes) the collection. Useful for testing or starting fresh.
        """
        self.client.delete_collection(name=self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            ),
        )
        print(f"Collection '{self.collection.name}' has been reset.")
