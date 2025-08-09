import tiktoken
from typing import List


class TextChunker:
    """
    Splits raw text into manageable, overlapping chunks suitable for embedding models.
    Uses tiktoken for accurate token counting.
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initializes the TextChunker with a specific tiktoken encoding.

        Args:
            encoding_name (str): The name of the tiktoken encoding to use.
                                 Defaults to "cl100k_base" (used by OpenAI models).
        """
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Splits text into chunks based on token count with a sliding window overlap.

        Args:
            text (str): The input text to chunk.
            chunk_size (int): The maximum number of tokens per chunk.
            chunk_overlap (int): The number of tokens to overlap between consecutive chunks.

        Returns:
            List[str]: A list of text chunks.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size.")

        tokens = self.tokenizer.encode(text)
        chunks = []

        if not tokens:
            return []

        start_index = 0
        while start_index < len(tokens):
            end_index = min(start_index + chunk_size, len(tokens))
            chunk_tokens = tokens[start_index:end_index]
            chunks.append(self.tokenizer.decode(chunk_tokens))

            if end_index == len(tokens):
                break

            start_index += chunk_size - chunk_overlap

            if start_index >= len(tokens) - chunk_overlap and end_index == len(tokens):
                break

        return chunks
