import chromadb
from chromadb.config import Settings

import logging

from typing import List, Dict, Any, Optional, Sequence, Union, Mapping

# Get a logger for this module
logger = logging.getLogger(__name__)


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
        embeddings: List[Sequence[float]],
        metadatas: List[Mapping[str, Optional[Union[str, int, float, bool]]]],
        ids: Optional[List[str]] = None,
    ):
        if not ids:
            current_count = self.collection.count()
            ids = [f"doc_{current_count + i}" for i in range(len(documents))]

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
        logger.info(
            f"Added {len(documents)} documents to ChromaDB collection '{self.collection.name}'."
        )

    def query(
        self, query_embedding: Sequence[float], num_results: int = 5
    ) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=num_results,
            include=["documents", "metadatas", "distances"],
        )

        formatted_results = []

        # Ensure all expected keys exist and their first elements are not None
        if (
            results
            and results.get("documents")
            and results.get("metadatas")
            and results.get("distances")
            and results["documents"]
            and results["metadatas"]
            and results["distances"]
        ):
            # Iterate based on the length of the documents list, as it's the primary data
            # and assume metadatas and distances will have corresponding entries.
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
        logger.info(f"Collection '{self.collection.name}' has been reset.")

    def print_all_documents_table(self):
        """
        Prints a table of all records in the collection for debugging purposes.
        The document content is truncated to the first 20 characters.
        """
        all_records = self.collection.get(include=["documents", "metadatas"])

        if (
            not all_records
            or not all_records.get("ids")
            or not all_records.get("documents")
            or not all_records.get("metadatas")
        ):
            logger.info("No documents found in the collection.")
            return

        logger.info("\n--- All Documents in Collection ---")
        logger.info(f"{'ID':<10} | {'Document (first 20 chars)':<25} | {'Metadata'}")
        logger.info("-" * 70)

        documents = all_records.get("documents")
        metadatas = all_records.get("metadatas")

        for i in range(len(all_records["ids"])):
            doc_id = all_records["ids"][i]
            document = "" if documents is None else documents[i]
            metadata = "" if metadatas is None else metadatas[i]

            truncated_document = (
                document[:20] + "..." if len(document) > 20 else document
            )
            logger.info(f"{doc_id:<10} | {truncated_document:<25} | {metadata}")
        logger.info("-----------------------------------\n")
