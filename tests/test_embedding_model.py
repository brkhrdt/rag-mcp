# tests/test_embedding_model.py
import pytest
from rag_mcp.embedding_model import EmbeddingModel
from typing import List

# Use a smaller, faster model for testing if possible, or mock the model.
# For simplicity, we'll use the default, but be aware it downloads on first run.
# If tests are slow, consider mocking sentence_transformers.SentenceTransformer.


@pytest.fixture(scope="module")
def embedding_model():
    """Provides an EmbeddingModel instance for tests."""
    # Using a very small model for faster tests.
    # 'all-MiniLM-L6-v2' is still relatively small and good for general use.
    # For CI/CD, consider 'all-MiniLM-L6-v2' or even smaller if available.
    return EmbeddingModel(model_name="all-MiniLM-L6-v2")


def test_embed_single_string(embedding_model):
    """Tests embedding a single string."""
    text = "Hello, world!"
    embedding = embedding_model.embed(text)
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)


def test_embed_list_of_strings(embedding_model):
    """Tests embedding a list of strings."""
    texts = ["First sentence.", "Second sentence.", "Third sentence."]
    embeddings = embedding_model.embed(texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    for emb in embeddings:
        assert isinstance(emb, list)
        assert len(emb) > 0
        assert all(isinstance(x, float) for x in emb)


def test_embed_empty_string(embedding_model):
    """Tests embedding an empty string."""
    text = ""
    embedding = embedding_model.embed(text)
    assert isinstance(embedding, list)
    # An empty string might still produce an embedding vector depending on the model
    # The important part is that it doesn't raise an error and returns a valid type.
    assert (
        len(embedding) > 0
    )  # Sentence-transformers usually returns a vector for empty string


def test_embed_empty_list(embedding_model):
    """Tests embedding an empty list of strings."""
    texts: List[str] = []
    embeddings = embedding_model.embed(texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 0


def test_embed_invalid_input_type(embedding_model):
    """Tests embedding with an invalid input type."""
    with pytest.raises(
        TypeError, match="Input 'texts' must be a string or a list of strings."
    ):
        embedding_model.embed(123)  # type: ignore
    with pytest.raises(
        TypeError, match="Input 'texts' must be a string or a list of strings."
    ):
        embedding_model.embed({"key": "value"})  # type: ignore
