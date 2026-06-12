# services/ranking/cosine_similarity.py
"""
Cosine similarity calculations for vector comparison.
"""

import math
from typing import Dict


class CosineSimilarity:
    """
    Handles cosine similarity calculations.
    
    Formula: cos(q,d) = (q·d) / (||q|| × ||d||)
    """
    
    @staticmethod
    def compute(
        query_vector: Dict[str, float], 
        doc_vector: Dict[str, float]
    ) -> float:
        """
        Compute cosine similarity between query and document vectors.
        
        Args:
            query_vector: Query vector (term -> weight)
            doc_vector: Document vector (term -> weight)
            
        Returns:
            Cosine similarity between 0 and 1
        """
        dot_product = CosineSimilarity._dot_product(query_vector, doc_vector)
        
        query_norm = CosineSimilarity._norm(query_vector)
        doc_norm = CosineSimilarity._norm(doc_vector)
        
        if query_norm == 0 or doc_norm == 0:
            return 0.0
        
        return dot_product / (query_norm * doc_norm)
    
    @staticmethod
    def _dot_product(
        v1: Dict[str, float], 
        v2: Dict[str, float]
    ) -> float:
        """
        Compute dot product between two vectors.
        
        Formula: Σ(v1_i × v2_i)
        """
        return sum(
            weight * v2.get(term, 0) 
            for term, weight in v1.items()
        )
    
    @staticmethod
    def _norm(vector: Dict[str, float]) -> float:
        """
        Compute Euclidean norm of a vector.
        
        Formula: √(Σ(weight_i²))
        """
        return math.sqrt(sum(weight ** 2 for weight in vector.values()))
    
    @staticmethod
    def batch_compute(
        query_vector: Dict[str, float],
        doc_vectors: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Compute similarity scores for multiple documents.
        
        Args:
            query_vector: Query vector
            doc_vectors: Dictionary mapping doc_id to document vector
            
        Returns:
            Dictionary mapping doc_id to similarity score
        """
        scores = {}
        
        for doc_id, doc_vector in doc_vectors.items():
            score = CosineSimilarity.compute(query_vector, doc_vector)
            if score > 0:
                scores[doc_id] = score
        
        return scores