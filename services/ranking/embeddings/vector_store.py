# services/ranking/embeddings/vector_store.py
"""
FAISS-based vector store for fast dense retrieval.
Responsible ONLY for storing vectors and finding nearest neighbors.
"""

import os
import pickle
from typing import List, Tuple, Optional

import numpy as np
import faiss


class VectorStore:
    """
    Stores document vectors in a FAISS index for fast similarity search.

    Uses Inner Product (dot product) search since vectors are L2-normalized,
    which equals cosine similarity for normalized vectors.

    Usage:
        store = VectorStore(dim=384)
        store.add_documents(doc_ids, vectors)
        results = store.search(query_vector, top_k=10)
        store.save("data/index/vector_store.faiss")
    """

    def __init__(self, dim: int):
        """
        Initialize the vector store.

        Args:
            dim: Dimension of the vectors (must match embedding model output)
        """
        self.dim = dim
        self._doc_ids: List[str] = []          # maps FAISS index position → doc_id
        self._index = faiss.IndexFlatIP(dim)   # Inner Product = cosine for normalized vecs
        print(f"✅ VectorStore initialized — dim={dim}")

    def add_documents(self, doc_ids: List[str], vectors: np.ndarray) -> None:
        """
        Add documents to the vector store.

        Args:
            doc_ids: List of document ID strings
            vectors: numpy array of shape (n_docs, dim), L2-normalized
        """
        if len(doc_ids) != vectors.shape[0]:
            raise ValueError(
                f"Mismatch: {len(doc_ids)} doc_ids but {vectors.shape[0]} vectors"
            )

        vectors = vectors.astype(np.float32)
        self._index.add(vectors)
        self._doc_ids.extend(doc_ids)
        print(f"✅ Added {len(doc_ids)} vectors — total: {self.total}")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find the most similar documents to a query vector.

        Args:
            query_vector: 1D numpy array of shape (dim,)
            top_k: Number of results to return

        Returns:
            List of (doc_id, score) tuples, sorted by score descending
        """
        if self.total == 0:
            return []

        # FAISS needs 2D input: shape (1, dim)
        query = query_vector.astype(np.float32).reshape(1, -1)
        top_k = min(top_k, self.total)

        scores, indices = self._index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:           # FAISS returns -1 for empty slots
                continue
            doc_id = self._doc_ids[idx]
            results.append((doc_id, float(score)))

        return results

    def save(self, path: str) -> None:
        """
        Save the vector store to disk.

        Args:
            path: File path (e.g. "data/index/vector_store.faiss")
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save FAISS index
        faiss.write_index(self._index, path)

        # Save doc_ids alongside
        ids_path = path.replace(".faiss", "_ids.pkl")
        with open(ids_path, "wb") as f:
            pickle.dump(self._doc_ids, f)

        print(f"✅ VectorStore saved → {path}")
        print(f"   doc_ids  → {ids_path}")

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        """
        Load a previously saved vector store from disk.

        Args:
            path: Path to the .faiss file

        Returns:
            Loaded VectorStore instance
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"VectorStore not found: {path}")

        # Load FAISS index
        index = faiss.read_index(path)
        dim = index.d

        # Load doc_ids
        ids_path = path.replace(".faiss", "_ids.pkl")
        with open(ids_path, "rb") as f:
            doc_ids = pickle.load(f)

        # Reconstruct instance
        store = cls(dim=dim)
        store._index = index
        store._doc_ids = doc_ids

        print(f"✅ VectorStore loaded — {len(doc_ids)} vectors, dim={dim}")
        return store

    @property
    def total(self) -> int:
        """Total number of vectors stored."""
        return self._index.ntotal

    def __len__(self) -> int:
        return self.total

    def __repr__(self) -> str:
        return f"VectorStore(total={self.total}, dim={self.dim})"