# services/ranking/bm25_scorer.py
import math
from typing import List

class BM25Scorer:
    """
    Pure mathematical scorer for BM25.
    Contains no loading, index-building, or preprocessing dependencies.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def compute_idf(self, df: int, total_docs: int) -> float:
        """
        Compute Inverse Document Frequency using the BM25 formula.
        """
        numerator = total_docs - df + 0.5
        denominator = df + 0.5
        # Ensure idf is non-negative and avoid math domain error with epsilon
        return max(math.log(numerator / denominator + 1e-10), 0.0)

    def score_term(self, tf: int, doc_len: int, avg_doc_len: float, idf: float) -> float:
        """
        Compute BM25 score for a single term in a document.
        """
        if tf <= 0 or idf <= 0:
            return 0.0
        
        # Length normalization factor
        length_norm = (1 - self.b) + self.b * (doc_len / avg_doc_len)
        
        # Term frequency component with saturation
        tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * length_norm)
        
        return idf * tf_component