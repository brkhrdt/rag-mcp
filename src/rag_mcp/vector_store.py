import chromadb
from chromadb.config import Settings


from typing import List, Dict, Any, Optional


class VectorStore:
    def __init__(
        self,
        collection_name: str = "rag_collection",
        persist_directory: str = "chroma_db",
    ):
        self.client = chromadb.PersistentClient(
            path=persist_directory, settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
        )

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ):
        if not ids:
            ids = [f"doc_{i}" for i in range(len(documents))]

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
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=num_results,
            include=["documents", "metadatas", "distances"],
        )

        formatted_results = []

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
        self.client.delete_collection(name=self.collection.name)

        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
        )
        print(f"Collection '{self.collection.name}' has been reset.")
