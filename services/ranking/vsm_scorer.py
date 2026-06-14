# services/ranking/vsm_scorer.py
import math
from collections import defaultdict
from typing import Dict, List

class VSMScorer:
    """
    Pure mathematical scorer for the Vector Space Model (VSM).
    Handles TF-IDF calculations and Cosine Similarity.
    """
    @staticmethod
    def compute_tf(tokens: List[str], normalize: bool = True) -> Dict[str, float]:
        """
        Compute Term Frequency.
        """
        if not tokens:
            return {}
        
        term_counts = defaultdict(int)
        for term in tokens:
            term_counts[term] += 1
            
        if normalize:
            doc_length = len(tokens)
            return {term: count / doc_length for term, count in term_counts.items()}
        
        return dict(term_counts)

    @staticmethod
    def compute_idf(doc_frequency: Dict[str, int], total_docs: int, smoothing: bool = True) -> Dict[str, float]:
        """
        Compute Inverse Document Frequency.
        """
        idf_values = {}
        for term, doc_count in doc_frequency.items():
            if smoothing:
                # Smoothed IDF: log10((N+1)/(df+1)) + 1
                idf = math.log10((total_docs + 1) / (doc_count + 1)) + 1
            else:
                # Standard IDF: log10(N/df)
                idf = math.log10(total_docs / doc_count) if doc_count > 0 else 0.0
            
            idf_values[term] = idf
        return idf_values

    @staticmethod
    def compute_tfidf(tf: Dict[str, float], idf: Dict[str, float]) -> Dict[str, float]:
        """
        Compute TF-IDF weights.
        """
        return {
            term: tf_value * idf.get(term, 0.0)
            for term, tf_value in tf.items()
        }

    @staticmethod
    def cosine_similarity(v1: Dict[str, float], v2: Dict[str, float]) -> float:
        """
        Compute Cosine Similarity between two sparse vectors.
        """
        dot_product = sum(weight * v2.get(term, 0.0) for term, weight in v1.items())
        norm1 = math.sqrt(sum(weight ** 2 for weight in v1.values()))
        norm2 = math.sqrt(sum(weight ** 2 for weight in v2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)