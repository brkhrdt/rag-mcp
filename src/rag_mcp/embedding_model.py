import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union


class EmbeddingModel:
    """
    Converts text chunks and queries into numerical vector embeddings.
    Wraps a sentence-transformers model and supports GPU acceleration.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed(
        self, texts: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """
        Generates embeddings for the given text or list of texts.

        Args:
            texts: A single string or a list of strings to embed.

        Returns:
            A list of floats (for a single string) or a list of lists of floats
            (for a list of strings), representing the embeddings.
        """
        if isinstance(texts, str):
            return self.model.encode(texts).tolist()
        elif isinstance(texts, list):
            return self.model.encode(texts).tolist()
        else:
            raise TypeError("Input 'texts' must be a string or a list of strings.")
            return self.model.encode(texts).tolist()

    @property
    def max_input_tokens(self) -> int:
        """Returns the maximum number of tokens the model can process."""

        return self.model.max_seq_length
