# services/ranking/embeddings/embedding_model.py
"""
Embedding model wrapper using Sentence-BERT.
Responsible ONLY for loading the model and encoding text into vectors.
"""

from typing import List, Union
import numpy as np


class EmbeddingModel:
    """
    Wraps a Sentence-BERT model for encoding text into dense vectors.

    Supports any model from sentence-transformers library.
    Default: all-MiniLM-L6-v2 (fast, small, good quality — ideal for this project).

    Usage:
        model = EmbeddingModel()
        vector = model.encode("cloud storage solutions")
        vectors = model.encode_batch(["text one", "text two"])
    """

    # Available model options — choose based on speed vs quality tradeoff
    MODELS = {
        "fast":    "all-MiniLM-L6-v2",        # 80MB,  384-dim, fastest
        "balanced": "all-mpnet-base-v2",        # 420MB, 768-dim, best quality
        "clinical": "pritamdeka/S-PubMedBert-MS-MARCO", # 420MB, 768-dim, biomedical
        "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",  # for Arabic too
    }


    def __init__(self, model_key: str = "fast"):
        """
        Initialize the embedding model.

        Args:
            model_key: One of "fast", "balanced", "multilingual"
                       or any valid sentence-transformers model name.
        """
        from sentence_transformers import SentenceTransformer

        model_name = self.MODELS.get(model_key, model_key)
        print(f"🤖 Loading embedding model: {model_name}")
        self._model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.vector_dim = self._model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded — vector dimension: {self.vector_dim}")

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single text string into a dense vector.

        Args:
            text: Input text string

        Returns:
            numpy array of shape (vector_dim,)
        """
        if not text or not isinstance(text, str):
            return np.zeros(self.vector_dim, dtype=np.float32)

        vector = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vector.astype(np.float32)

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode a list of texts into a matrix of dense vectors.

        Args:
            texts: List of text strings
            batch_size: Number of texts to encode at once
            show_progress: Show tqdm progress bar

        Returns:
            numpy array of shape (len(texts), vector_dim)
        """
        if not texts:
            return np.zeros((0, self.vector_dim), dtype=np.float32)

        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True   # L2-normalize → cosine similarity = dot product
        )
        return vectors.astype(np.float32)

    @property
    def dim(self) -> int:
        """Vector dimension."""
        return self.vector_dim