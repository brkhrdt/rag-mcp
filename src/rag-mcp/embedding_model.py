# src/rag-mcp/embedding_model.py
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Union

class EmbeddingModel:
    """
    Converts text chunks and queries into numerical vector embeddings.
    Wraps a sentence-transformers model and supports GPU acceleration.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the EmbeddingModel.

        Args:
            model_name (str): The name of the sentence-transformers model to load.
                              Defaults to "all-MiniLM-L6-v2".
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generates embeddings for a single text or a list of texts.

        Args:
            texts (Union[str, List[str]]): The text or list of texts to embed.

        Returns:
            Union[List[float], List[List[float]]]: A list of floats for a single text,
                                                   or a list of lists of floats for multiple texts.
        """
        if isinstance(texts, str):
            return self.model.encode(texts).tolist()
        elif isinstance(texts, list):
            return self.model.encode(texts).tolist()
        else:
            raise TypeError("Input 'texts' must be a string or a list of strings.")

