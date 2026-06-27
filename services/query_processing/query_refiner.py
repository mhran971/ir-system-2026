# services/query_processing/query_refiner.py
"""
Query Refinement Module with Multiple Strategies
"""

import re
from typing import List, Dict, Set, Optional, Tuple
from collections import Counter
import math


class QueryRefiner:
    """
    Query refinement with multiple strategies:
    - Spelling correction (customizable)
    - PRF with term selection
    - Domain-specific term filtering
    """
    
    def __init__(
        self,
        known_terms: Set[str],
        term_frequencies: Dict[str, int],
        total_docs: int,
        medical_terms: Optional[Set[str]] = None
    ):
        """
        Args:
            known_terms: All terms in the index
            term_frequencies: Document frequency per term
            total_docs: Total number of documents
            medical_terms: Optional set of medical domain terms
        """
        self.known_terms = known_terms
        self.term_frequencies = term_frequencies
        self.total_docs = total_docs
        self.medical_terms = medical_terms or set()
        
        # Create a set of common stopwords
        self.stopwords = {
            'a', 'an', 'the', 'of', 'to', 'for', 'with', 'on', 'at', 'from',
            'by', 'in', 'as', 'is', 'was', 'were', 'are', 'am', 'be', 'been',
            'being', 'and', 'or', 'but', 'so', 'for', 'nor', 'yet'
        }
    
    def refine(
        self,
        query: str,
        strategy: str = "conservative",  # "conservative", "moderate", "aggressive"
        apply_spelling: bool = True,
        apply_prf: bool = True,
        top_docs: Optional[List[Dict]] = None,
        num_prf_terms: int = 2,
        min_term_freq: int = 1,
        max_term_freq_ratio: float = 0.3,
        min_idf_threshold: float = 1.0,
    ) -> Dict[str, any]:
        """
        Refine a query with controlled strategies.
        
        Args:
            strategy: "conservative", "moderate", "aggressive"
            apply_spelling: Whether to apply spelling correction
            apply_prf: Whether to apply PRF
            top_docs: Top documents from initial retrieval
            num_prf_terms: Number of PRF terms to add (capped by strategy)
            min_term_freq: Minimum document frequency for PRF terms
            max_term_freq_ratio: Maximum document frequency ratio (avoid common terms)
            min_idf_threshold: Minimum IDF for PRF terms
        
        Returns:
            Dict with original_query, expanded_query, added_terms, spelling_corrections
        """
        original_query = query.strip()
        expanded_query = original_query
        added_terms = []
        spelling_corrections = {}
        
        # 1. Spelling Correction
        if apply_spelling:
            expanded_query, spelling_corrections = self._correct_spelling(expanded_query)
        
        # 2. PRF Expansion (with term selection)
        if apply_prf and top_docs and len(top_docs) > 0:
            prf_terms = self._extract_prf_terms(
                top_docs=top_docs,
                num_terms=num_prf_terms,
                min_term_freq=min_term_freq,
                max_term_freq_ratio=max_term_freq_ratio,
                min_idf_threshold=min_idf_threshold,
                strategy=strategy,
            )
            
            # Add terms that are not already in query
            query_terms = set(expanded_query.lower().split())
            for term in prf_terms:
                if term not in query_terms and len(term) > 2:
                    expanded_query += f" {term}"
                    added_terms.append(term)
                    query_terms.add(term)
        
        return {
            "original_query": original_query,
            "expanded_query": expanded_query,
            "added_terms": added_terms,
            "spelling_corrections": spelling_corrections,
        }
    
    def _correct_spelling(self, query: str) -> Tuple[str, Dict[str, str]]:
        """
        Simple spelling correction using known_terms.
        For production, use a domain-specific spell checker.
        """
        words = query.split()
        corrections = {}
        corrected_words = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in self.known_terms:
                corrected_words.append(word)
            else:
                # Try to find a correction
                correction = self._find_closest_term(word_lower)
                if correction and correction != word_lower:
                    corrections[word] = correction
                    corrected_words.append(correction)
                else:
                    corrected_words.append(word)
        
        return " ".join(corrected_words), corrections
    
    def _find_closest_term(self, word: str, max_distance: int = 2) -> Optional[str]:
        """
        Find closest term using Levenshtein distance.
        Only consider terms that are not too far.
        """
        # Simple Levenshtein distance implementation
        def levenshtein(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return levenshtein(s2, s1)
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        # Only search if word is at least 3 characters
        if len(word) < 3:
            return None
        
        # Search for close terms
        best_match = None
        best_distance = float('inf')
        
        # Limit search to terms that start with same letter for speed
        candidates = [t for t in self.known_terms if t.startswith(word[0])]
        
        for term in candidates:
            if len(term) < 3:
                continue
            dist = levenshtein(word, term)
            if dist < best_distance and dist <= max_distance:
                best_distance = dist
                best_match = term
        
        return best_match
    
    def _extract_prf_terms(
        self,
        top_docs: List[Dict],
        num_terms: int = 2,
        min_term_freq: int = 1,
        max_term_freq_ratio: float = 0.3,
        min_idf_threshold: float = 1.0,
        strategy: str = "conservative",
    ) -> List[str]:
        """
        Extract PRF terms with intelligent filtering.
        
        Strategy-specific behavior:
        - conservative: Only add very rare, domain-specific terms
        - moderate: Add terms with high IDF, avoid common terms
        - aggressive: Add more terms, less filtering
        """
        # Adjust parameters based on strategy
        if strategy == "conservative":
            num_terms = min(num_terms, 1)
            max_term_freq_ratio = 0.1
            min_idf_threshold = 2.0
        elif strategy == "moderate":
            num_terms = min(num_terms, 2)
            max_term_freq_ratio = 0.2
            min_idf_threshold = 1.5
        elif strategy == "aggressive":
            num_terms = min(num_terms, 4)
            max_term_freq_ratio = 0.4
            min_idf_threshold = 0.8
        
        # Collect terms from top documents
        term_scores = {}
        
        for doc in top_docs:
            doc_id = doc.get("doc_id", "")
            doc_text = self._get_doc_text(doc_id)
            if not doc_text:
                continue
            
            # Tokenize and count terms
            terms = self._tokenize(doc_text)
            term_counts = Counter(terms)
            
            for term, count in term_counts.items():
                if term in self.stopwords:
                    continue
                if len(term) < 3:
                    continue
                
                # IDF weight
                df = self.term_frequencies.get(term, 0)
                idf = math.log((self.total_docs + 1) / (df + 1)) if df > 0 else 0
                
                # Frequency ratio
                freq_ratio = df / self.total_docs if self.total_docs > 0 else 1.0
                
                # Score: term frequency * IDF
                score = count * idf
                
                # Apply filters
                if df < min_term_freq:
                    continue
                if freq_ratio > max_term_freq_ratio:
                    continue
                if idf < min_idf_threshold:
                    continue
                
                # Bonus for medical terms
                if term in self.medical_terms:
                    score *= 1.5
                
                if term not in term_scores or score > term_scores[term]:
                    term_scores[term] = score
        
        # Sort by score and return top terms
        sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
        return [term for term, score in sorted_terms[:num_terms]]
    
    def _get_doc_text(self, doc_id: str) -> str:
        """Get document text by doc_id - overridden in evaluation."""
        # This will be overridden when instantiated
        return ""
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer."""
        # Convert to lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'[a-z0-9]+', text)
        return tokens