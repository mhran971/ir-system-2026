# services/ranking/bm25_scorer.py
import math
from typing import Dict, List, Optional

class BM25Scorer:
    """
    Pure mathematical scorer for BM25.
    Contains no loading, index-building, or preprocessing dependencies.
    
    BM25 Formula:
    score(q,d) = Σ IDF(qi) × (tf(qi,d) × (k1+1)) / (tf(qi,d) + k1 × (1-b + b × |d|/avgdl))
    
    Parameters:
    - k1: Term frequency saturation (1.2 - 2.0, default: 1.5)
    - b: Length normalization (0.0 - 1.0, default: 0.75)
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        if k1 <= 0:
            raise ValueError("k1 must be > 0")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        
        self.k1 = k1
        self.b = b

    def compute_idf(self, df: int, total_docs: int) -> float:
        """
        Compute Inverse Document Frequency using the BM25 formula.
        
        Formula: IDF = log((N - df + 0.5) / (df + 0.5))
        """
        if df <= 0 or total_docs <= 0:
            return 0.0
        
        numerator = total_docs - df + 0.5
        denominator = df + 0.5
        
        if numerator <= 0 or denominator <= 0:
            return 0.0
        
        # ✅ التصحيح: إزالة return المكرر
        idf = math.log(numerator / denominator)
        return max(idf, 0.0)  # تأكد من عدم وجود قيم سالبة

    def score_term(self, tf: int, doc_len: int, avg_doc_len: float, idf: float) -> float:
        """
        Compute BM25 score for a single term in a document.
        """
        if tf <= 0 or idf <= 0 or avg_doc_len <= 0:
            return 0.0
        
        # Length normalization factor
        length_norm = (1 - self.b) + self.b * (doc_len / avg_doc_len)
        
        # Term frequency component with saturation
        denominator = tf + self.k1 * length_norm
        if denominator == 0:
            return 0.0
        
        tf_component = (tf * (self.k1 + 1)) / denominator
        return idf * tf_component
    
    def score_document(
        self,
        doc_tokens: List[str],
        query_tokens: List[str],
        doc_frequency: Dict[str, int],
        total_docs: int,
        avg_doc_len: float
    ) -> float:
        """
        Compute total BM25 score for a document given a query.
        """
        if not doc_tokens or not query_tokens:
            return 0.0
        
        doc_len = len(doc_tokens)
        
        # Count term frequencies in the document
        term_counts = {}
        for term in doc_tokens:
            term_counts[term] = term_counts.get(term, 0) + 1
        
        total_score = 0.0
        seen_terms = set()
        
        for term in query_tokens:
            if term in seen_terms:
                continue
            seen_terms.add(term)
            
            tf = term_counts.get(term, 0)
            df = doc_frequency.get(term, 0)
            
            idf = self.compute_idf(df, total_docs)
            score = self.score_term(tf, doc_len, avg_doc_len, idf)
            
            total_score += score
        
        return total_score
    
    def get_parameter_info(self) -> Dict[str, Dict[str, str]]:
        """Return explanation of BM25 parameters."""
        return {
            'k1': {
                'name': 'Term Frequency Saturation',
                'formula': f'k1 = {self.k1}',
                'range': '1.2 - 2.0',
                'description': 'Controls how quickly TF saturates. Higher values give more weight to TF.'
            },
            'b': {
                'name': 'Length Normalization',
                'formula': f'b = {self.b}',
                'range': '0.0 - 1.0',
                'description': 'Controls length normalization. 0 = no normalization, 1 = full normalization.'
            }
        }