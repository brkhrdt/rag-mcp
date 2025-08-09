import tiktoken
from typing import List


class TextChunker:
    def __init__(self, encoding_name: str = "cl100k_base"):
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
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
