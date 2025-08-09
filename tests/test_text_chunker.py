# tests/test_text_chunker.py
import pytest
from src.rag_mcp.text_chunker import TextChunker


@pytest.fixture
def text_chunker():
    """Provides a TextChunker instance for tests."""
    return TextChunker(encoding_name="cl100k_base")  # Use a common encoding


def test_chunk_text_basic(text_chunker):
    """Tests basic chunking without overlap."""
    text = "This is a short sentence. This is another short sentence. And a third one."
    # Using cl100k_base:
    # "This is a short sentence." -> 6 tokens
    # " This is another short sentence." -> 7 tokens
    # " And a third one." -> 5 tokens
    # Total: 18 tokens
    chunks = text_chunker.chunk_text(text, chunk_size=10, chunk_overlap=0)
    assert len(chunks) == 2  # 18 tokens, chunk_size 10 -> 2 chunks (10 + 8)
    assert "This is a short sentence. This is another" in chunks[0]
    assert (
        "short sentence. And a third one." in chunks[1]
    )  # The second chunk starts from where the first ended


def test_chunk_text_with_overlap(text_chunker):
    """Tests chunking with overlap."""
    text = "This is a very long text that needs to be chunked with overlap for proper retrieval."
    # "This is a very long text that needs to be chunked with overlap for proper retrieval."
    # cl100k_base: 23 tokens
    chunks = text_chunker.chunk_text(text, chunk_size=15, chunk_overlap=5)
    assert len(chunks) == 2  # (23 - 5) / (15 - 5) = 1.8 -> 2 chunks
    # Chunk 1: first 15 tokens
    # Chunk 2: starts at token 15-5=10, goes to end
    assert len(text_chunker.tokenizer.encode(chunks[0])) <= 15
    assert len(text_chunker.tokenizer.encode(chunks[1])) <= 15
    # Check for overlap content
    # This is a bit tricky to assert precisely without knowing the exact tokenization.
    # A simpler check is that the total length is reasonable and chunks are not identical.
    assert chunks[0] != chunks[1]
    assert (
        text_chunker.tokenizer.decode(text_chunker.tokenizer.encode(text)[10:15])
        in chunks[0]
    )  # Part of overlap in first chunk
    assert (
        text_chunker.tokenizer.decode(text_chunker.tokenizer.encode(text)[10:15])
        in chunks[1]
    )  # Part of overlap in second chunk


def test_chunk_text_smaller_than_chunk_size(text_chunker):
    """Tests text smaller than chunk size."""
    text = "A very short text."  # 5 tokens
    chunks = text_chunker.chunk_text(text, chunk_size=10, chunk_overlap=0)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_empty(text_chunker):
    """Tests empty text input."""
    chunks = text_chunker.chunk_text("", chunk_size=10, chunk_overlap=0)
    assert len(chunks) == 0


def test_chunk_text_overlap_greater_than_chunk_size(text_chunker):
    """Tests invalid overlap value."""
    text = "Some text."
    with pytest.raises(ValueError, match="Chunk overlap must be less than chunk size."):
        text_chunker.chunk_text(text, chunk_size=5, chunk_overlap=5)
    with pytest.raises(ValueError, match="Chunk overlap must be less than chunk size."):
        text_chunker.chunk_text(text, chunk_size=5, chunk_overlap=6)


def test_chunk_text_exact_fit(text_chunker):
    """Tests text that fits exactly into chunks."""
    # Let's make it simpler for exact fit:
    text_exact = "token one token two token three token four token five token six"
    # 12 tokens (cl100k_base)
    chunks = text_chunker.chunk_text(text_exact, chunk_size=6, chunk_overlap=0)
    assert len(chunks) == 2
    assert (
        text_chunker.tokenizer.decode(text_chunker.tokenizer.encode(text_exact)[:6])
        == chunks[0]
    )
    assert (
        text_chunker.tokenizer.decode(text_chunker.tokenizer.encode(text_exact)[6:])
        == chunks[1]
    )

