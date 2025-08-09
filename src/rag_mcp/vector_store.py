# src/rag-mcp/vector_store.py
import chromadb

# from chromadb.utils import embedding_functions # No longer needed if we always provide embeddings
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
        # We will explicitly provide embeddings when adding documents and querying.
        # Therefore, we don't need to set an embedding_function for the collection
        # if we are always providing them ourselves.
        # ChromaDB will use a default identity function if none is provided,
        # which is appropriate when external embeddings are always supplied.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            # embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            #     model_name="all-MiniLM-L6-v2"
            # ), # Removed as we provide embeddings externally
        )

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
            # Generate simple IDs if not provided. ChromaDB can also generate UUIDs by default.
            # For testing, explicit simple IDs are useful.
            ids = [f"doc_{i}" for i in range(len(documents))]

        # Ensure metadatas list is the same length as documents, filling with None if not provided
        if metadatas is None:
            metadatas = [None] * len(documents)

        if (
            len(documents) != len(embeddings)
            or len(documents) != len(metadatas)
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
        # Check if results and documents are not empty before iterating
        if results and results["documents"] and results["documents"][0]:
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
        # Re-create the collection without an embedding function, consistent with __init__
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            # embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            #     model_name="all-MiniLM-L6-v2"
            # ), # Removed
        )
        print(f"Collection '{self.collection.name}' has been reset.")
