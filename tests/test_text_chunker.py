# tests/test_text_chunker.py
import pytest
from rag_mcp.text_chunker import TextChunker


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
    
    # Decode the expected token ranges to compare with actual chunks
    tokens = text_chunker.tokenizer.encode(text)
    expected_chunk0 = text_chunker.tokenizer.decode(tokens[0:10])
    expected_chunk1 = text_chunker.tokenizer.decode(tokens[10:18]) # Remaining 8 tokens

    assert chunks[0] == expected_chunk0
    assert chunks[1] == expected_chunk1


def test_chunk_text_with_overlap(text_chunker):
    """Tests chunking with overlap."""
    text = "This is a very long text that needs to be chunked with overlap for proper retrieval."
    # "This is a very long text that needs to be chunked with overlap for proper retrieval."
    # cl100k_base: 23 tokens
    chunks = text_chunker.chunk_text(text, chunk_size=15, chunk_overlap=5)
    assert len(chunks) == 2  # (23 - 5) / (15 - 5) = 1.8 -> 2 chunks
    # Chunk 1: first 15 tokens
    # Chunk 2: starts at token 15-5=10, goes to end
    
    tokens = text_chunker.tokenizer.encode(text)
    
    # Verify chunk lengths are within limits
    assert len(text_chunker.tokenizer.encode(chunks[0])) <= 15
    assert len(text_chunker.tokenizer.encode(chunks[1])) <= 15
    
    # Verify the content of the chunks based on token indices
    expected_chunk0 = text_chunker.tokenizer.decode(tokens[0:15])
    expected_chunk1 = text_chunker.tokenizer.decode(tokens[10:23]) # Starts at 10 (15-5), goes to end

    assert chunks[0] == expected_chunk0
    assert chunks[1] == expected_chunk1
    
    # Explicitly check for overlap content
    overlap_segment = text_chunker.tokenizer.decode(tokens[10:15])
    assert overlap_segment in chunks[0]
    assert overlap_segment in chunks[1]


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

def test_chunk_text_single_token_chunk_size(text_chunker):
    """Tests chunking with chunk_size = 1."""
    text = "Hello world!" # 2 tokens
    chunks = text_chunker.chunk_text(text, chunk_size=1, chunk_overlap=0)
    assert len(chunks) == 2
    assert chunks[0] == "Hello"
    assert chunks[1] == " world!"

def test_chunk_text_overlap_one_token(text_chunker):
    """Tests chunking with chunk_overlap = 1."""
    text = "This is a test sentence." # 6 tokens
    chunks = text_chunker.chunk_text(text, chunk_size=3, chunk_overlap=1)
    # Tokens: [This, is, a, test, sentence, .]
    # Chunk 1: [This, is, a] (3 tokens)
    # Start next at 3-1=2.
    # Chunk 2: [a, test, sentence] (3 tokens)
    # Start next at 2+3-1=4.
    # Chunk 3: [sentence, .] (2 tokens)
    assert len(chunks) == 3
    tokens = text_chunker.tokenizer.encode(text)
    assert chunks[0] == text_chunker.tokenizer.decode(tokens[0:3])
    assert chunks[1] == text_chunker.tokenizer.decode(tokens[2:5])
    assert chunks[2] == text_chunker.tokenizer.decode(tokens[4:6])
