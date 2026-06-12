# services/ranking/tfidf_calculator.py
"""
TF-IDF calculations for term weighting.
"""

import math
from collections import defaultdict
from typing import Dict, List


class TFIDFCalculator:
    """
    Handles TF-IDF calculations.
    
    Formulas:
    - TF(t,d) = count(t,d) / |d|
    - IDF(t) = log10((N+1)/(df(t)+1)) + 1
    - TF-IDF(t,d) = TF(t,d) × IDF(t)
    """
    
    @staticmethod
    def compute_tf(tokens: List[str], normalize: bool = True) -> Dict[str, float]:
        """
        Compute Term Frequency.
        
        Args:
            tokens: List of tokens from document/query
            normalize: If True, normalize by document length
            
        Returns:
            Dictionary mapping term to TF value
        """
        if not tokens:
            return {}
        
        # Count term frequencies
        term_counts = defaultdict(int)
        for term in tokens:
            term_counts[term] += 1
        
        # Normalize if requested
        if normalize:
            doc_length = len(tokens)
            return {
                term: count / doc_length 
                for term, count in term_counts.items()
            }
        
        return dict(term_counts)
    
    @staticmethod
    def compute_idf(
        doc_frequency: Dict[str, int], 
        total_docs: int,
        smoothing: bool = True
    ) -> Dict[str, float]:
        """
        Compute Inverse Document Frequency.
        
        Args:
            doc_frequency: Dictionary of term -> number of documents containing it
            total_docs: Total number of documents
            smoothing: If True, use smoothed version (+1 in numerator and denominator)
            
        Returns:
            Dictionary mapping term to IDF value
        """
        idf_values = {}
        
        for term, doc_count in doc_frequency.items():
            if smoothing:
                # Smoothed IDF: log10((N+1)/(df+1)) + 1
                idf = math.log10((total_docs + 1) / (doc_count + 1)) + 1
            else:
                # Standard IDF: log10(N/df)
                idf = math.log10(total_docs / doc_count) if doc_count > 0 else 0
            
            idf_values[term] = idf
        
        return idf_values
    
    @staticmethod
    def compute_tfidf(
        tf: Dict[str, float], 
        idf: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute TF-IDF weights.
        
        Args:
            tf: Term frequency dictionary
            idf: IDF dictionary
            
        Returns:
            Dictionary mapping term to TF-IDF weight
        """
        return {
            term: tf_value * idf.get(term, 0)
            for term, tf_value in tf.items()
        }
    
    @staticmethod
    def compute_term_details(
        term: str,
        tokens: List[str],
        idf: float
    ) -> Dict[str, float]:
        """
        Compute detailed TF, IDF, and TF-IDF for a term in a document.
        
        Args:
            term: The term to analyze
            tokens: Document tokens
            idf: IDF value for the term
            
        Returns:
            Dictionary with tf, idf, and tfidf values
        """
        result = {'tf': 0.0, 'idf': idf, 'tfidf': 0.0}
        
        if not tokens:
            return result
        
        # Calculate TF
        term_count = tokens.count(term)
        if term_count > 0:
            result['tf'] = term_count / len(tokens)
            result['tfidf'] = result['tf'] * idf
        
        return result