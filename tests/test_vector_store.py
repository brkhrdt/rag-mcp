# tests/test_vector_store.py
import pytest
from rag_mcp.vector_store import VectorStore
import shutil # Import shutil for rmtree


# Fixture for a temporary ChromaDB client and collection
@pytest.fixture
def temp_chroma_db(tmp_path):
    """Provides a temporary ChromaDB client and collection for testing."""
    persist_directory = tmp_path / "test_chroma_db"
    collection_name = "test_collection"
    # Ensure the directory is clean before starting
    if persist_directory.exists():
        shutil.rmtree(persist_directory)

    # Initialize VectorStore, which creates the client and collection
    vs = VectorStore(
        collection_name=collection_name, persist_directory=str(persist_directory)
    )
    yield vs
    # Teardown: Clean up the ChromaDB directory after tests
    if persist_directory.exists():
        shutil.rmtree(persist_directory)


def test_add_documents(temp_chroma_db):
    """Tests adding documents to the vector store."""
    documents = ["doc one", "doc two", "doc three"]
    embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]  # Dummy embeddings
    metadatas = [{"source": "file1"}, {"source": "file2"}, {"source": "file3"}]
    ids = ["id1", "id2", "id3"]

    temp_chroma_db.add_documents(documents, embeddings, metadatas, ids)

    # Verify documents were added by querying for them
    collection = temp_chroma_db.collection
    assert collection.count() == 3

    # Retrieve all to check content (less efficient but good for testing small sets)
    # ChromaDB's .get() method returns IDs by default, no need to include "ids" in the list.
    retrieved = collection.get(
        ids=["id1", "id2", "id3"], include=["documents", "metadatas"]
    )
    assert len(retrieved["documents"]) == 3
    assert "doc one" in retrieved["documents"]
    assert {"source": "file1"} in retrieved["metadatas"]
    assert set(retrieved["ids"]) == set(ids) # Verify IDs are also returned and match


def test_add_documents_no_ids_or_metadatas(temp_chroma_db):
    """Tests adding documents without providing IDs or metadatas."""
    documents = ["doc A", "doc B"]
    embeddings = [[0.7, 0.8], [0.9, 1.0]]

    temp_chroma_db.add_documents(documents, embeddings)
    collection = temp_chroma_db.collection
    assert collection.count() == 2
    # Check that default IDs were generated (e.g., "doc_0", "doc_1")
    # Remove "ids" from include as it's returned by default and causes an error if specified.
    retrieved = collection.get(include=["documents", "metadatas"])
    assert len(retrieved["ids"]) == 2
    assert "doc_0" in retrieved["ids"]
    assert "doc_1" in retrieved["ids"]
    # Metadatas should be None if not provided
    assert retrieved["metadatas"][0] is None
    assert retrieved["metadatas"][1] is None


def test_add_documents_mismatched_lengths(temp_chroma_db):
    """Tests adding documents with mismatched lengths of inputs."""
    documents = ["doc one", "doc two"]
    embeddings = [[0.1, 0.2]]  # Mismatched length
    with pytest.raises(
        ValueError,
        match="Lengths of documents, embeddings, metadatas, and ids must match.",
    ):
        temp_chroma_db.add_documents(documents, embeddings)


def test_query(temp_chroma_db):
    """Tests querying the vector store."""
    documents = [
        "apple pie is delicious",
        "banana split is tasty",
        "cherry tart is sweet",
    ]
    # Dummy embeddings, in a real scenario these would be from an actual model
    embeddings = [
        [0.1, 0.1, 0.1],  # apple
        [0.2, 0.2, 0.2],  # banana
        [0.3, 0.3, 0.3],  # cherry
    ]
    metadatas = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    ids = ["doc_a", "doc_b", "doc_c"]

    temp_chroma_db.add_documents(documents, embeddings, metadatas, ids)

    # Query for something similar to "apple"
    query_embedding = [0.11, 0.12, 0.13]  # Close to apple embedding

    results = temp_chroma_db.query(query_embedding, num_results=1)

    assert len(results) == 1
    assert results[0]["document"] == "apple pie is delicious"
    assert results[0]["metadata"] == {"id": "a"}
    assert "distance" in results[0]
    assert isinstance(results[0]["distance"], float)

    # Query for something similar to "banana"
    query_embedding_banana = [0.21, 0.22, 0.23]
    results_banana = temp_chroma_db.query(query_embedding_banana, num_results=1)
    assert results_banana[0]["document"] == "banana split is tasty"

    # Query with more results
    query_embedding_mixed = [0.15, 0.15, 0.15]  # Between apple and banana
    results_mixed = temp_chroma_db.query(query_embedding_mixed, num_results=3)
    assert len(results_mixed) == 3
    # Order might vary slightly based on exact distance, but all should be present
    assert set([r["document"] for r r in results_mixed]) == set(documents)


def test_query_no_results(temp_chroma_db):
    """Tests querying when no documents are in the store."""
    query_embedding = [0.1, 0.2, 0.3]
    results = temp_chroma_db.query(query_embedding, num_results=1)
    assert len(results) == 0


def test_reset(temp_chroma_db):
    """Tests resetting the vector store."""
    documents = ["test doc"]
    embeddings = [[0.1, 0.2]]
    temp_chroma_db.add_documents(documents, embeddings)
    assert temp_chroma_db.collection.count() == 1

    temp_chroma_db.reset()
    assert temp_chroma_db.collection.count() == 0

    # Ensure it's still functional after reset
    temp_chroma_db.add_documents(documents, embeddings)
    assert temp_chroma_db.collection.count() == 1
