import pytest
from rag_mcp.vector_store import VectorStore
import shutil


@pytest.fixture
def temp_chroma_db(tmp_path):
    """Provides a temporary ChromaDB client and collection for testing."""
    persist_directory = tmp_path / "test_chroma_db"
    collection_name = "test_collection"

    if persist_directory.exists():
        shutil.rmtree(persist_directory)

    vs = VectorStore(
        collection_name=collection_name, persist_directory=str(persist_directory)
    )
    yield vs

    if persist_directory.exists():
        shutil.rmtree(persist_directory)


def test_add_documents(temp_chroma_db):
    """Tests adding documents to the vector store."""
    documents = ["doc one", "doc two", "doc three"]
    embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    metadatas = [{"source": "file1"}, {"source": "file2"}, {"source": "file3"}]
    ids = ["id1", "id2", "id3"]

    temp_chroma_db.add_documents(documents, embeddings, metadatas, ids)

    collection = temp_chroma_db.collection
    assert collection.count() == 3

    retrieved = collection.get(
        ids=["id1", "id2", "id3"], include=["documents", "metadatas"]
    )
    assert len(retrieved["documents"]) == 3
    assert "doc one" in retrieved["documents"]
    assert {"source": "file1"} in retrieved["metadatas"]
    assert set(retrieved["ids"]) == set(ids)


def test_add_documents_no_ids_or_metadatas(temp_chroma_db):
    """Tests adding documents without providing IDs or metadatas."""
    documents = ["doc A", "doc B"]
    embeddings = [[0.7, 0.8], [0.9, 1.0]]
    metadatas = [None, None]

    temp_chroma_db.add_documents(documents, embeddings, metadatas)
    collection = temp_chroma_db.collection
    assert collection.count() == 2

    retrieved = collection.get(include=["documents", "metadatas"])
    assert len(retrieved["ids"]) == 2
    assert "doc_0" in retrieved["ids"]
    assert "doc_1" in retrieved["ids"]

    assert retrieved["metadatas"][0] is None
    assert retrieved["metadatas"][1] is None


def test_add_documents_mismatched_lengths(temp_chroma_db):
    """Tests adding documents with mismatched lengths of inputs."""
    documents = ["doc one", "doc two"]
    embeddings = [[0.1, 0.2]]
    metadatas = [None, None]
    with pytest.raises(
        ValueError,
        match="Lengths of documents, embeddings, metadatas, and ids must match.",
    ):
        temp_chroma_db.add_documents(documents, embeddings, metadatas)


def test_query(temp_chroma_db):
    """Tests querying the vector store."""
    documents = [
        "apple pie is delicious",
        "banana split is tasty",
        "cherry tart is sweet",
    ]

    embeddings = [
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.3, 0.3, 0.3],
    ]
    metadatas = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    ids = ["doc_a", "doc_b", "doc_c"]

    temp_chroma_db.add_documents(documents, embeddings, metadatas, ids)

    query_embedding = [0.11, 0.12, 0.13]

    results = temp_chroma_db.query(query_embedding, num_results=1)

    assert len(results) == 1
    assert results[0]["document"] == "apple pie is delicious"
    assert results[0]["metadata"] == {"id": "a"}
    assert "distance" in results[0]
    assert isinstance(results[0]["distance"], float)

    query_embedding_banana = [0.21, 0.22, 0.23]
    results_banana = temp_chroma_db.query(query_embedding_banana, num_results=1)
    assert results_banana[0]["document"] == "banana split is tasty"

    query_embedding_mixed = [0.15, 0.15, 0.15]
    results_mixed = temp_chroma_db.query(query_embedding_mixed, num_results=3)
    assert len(results_mixed) == 3

    assert set([r["document"] for r in results_mixed]) == set(documents)


def test_query_no_results(temp_chroma_db):
    """Tests querying when no documents are in the store."""
    query_embedding = [0.1, 0.2, 0.3]
    results = temp_chroma_db.query(query_embedding, num_results=1)
    assert len(results) == 0


def test_reset(temp_chroma_db):
    """Tests resetting the vector store."""
    documents = ["test doc"]
    embeddings = [[0.1, 0.2]]
    metadatas = [None]
    temp_chroma_db.add_documents(documents, embeddings, metadatas)
    assert temp_chroma_db.collection.count() == 1

    temp_chroma_db.reset()
    assert temp_chroma_db.collection.count() == 0

    temp_chroma_db.add_documents(documents, embeddings, metadatas)
    assert temp_chroma_db.collection.count() == 1
