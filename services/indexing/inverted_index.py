# services/indexing/inverted_index.py
"""
Inverted index for efficient document retrieval.
"""

from collections import defaultdict
from typing import Dict, List, Set, Optional, Any


class InvertedIndex:
    """
    Inverted index mapping terms to documents.
    
    Structure: term -> {doc_id: term_frequency}
    
    Example:
        >>> index = InvertedIndex()
        >>> index.add_document("doc1", ["cloud", "cloud", "storage"])
        >>> len(index)  # عدد المصطلحات الفريدة
        2
        >>> "cloud" in index  # التحقق من وجود مصطلح
        True
        >>> index["cloud"]  # الوصول المباشر
        {'doc1': 2}
    """
    
    def __init__(self):
        self._index: Dict[str, Dict[str, int]] = defaultdict(dict)
        self._doc_frequency: Dict[str, int] = defaultdict(int)
        self._doc_lengths: Dict[str, int] = {}
        self.N: int = 0
    
    def add_document(self, doc_id: str, tokens: List[str]) -> None:
        """
        Add a document to the inverted index.
        
        Args:
            doc_id: Document identifier
            tokens: List of preprocessed tokens
        """
        # Save document length
        self._doc_lengths[doc_id] = len(tokens)
        self.N += 1
        
        # Count term frequencies in document
        term_counts = defaultdict(int)
        for term in tokens:
            term_counts[term] += 1
        
        # Update index
        for term, count in term_counts.items():
            self._index[term][doc_id] = count
            self._doc_frequency[term] += 1

    def build(self, docs: List[Dict[str, Any]]) -> None:
        """
        Build the inverted index from a list of documents.
        """
        for doc in docs:
            doc_id = str(doc.get('doc_id', ''))
            tokens = doc.get('tokens', [])
            if doc_id:
                self.add_document(doc_id, tokens)

    def save(self, path: str = 'data/index/inverted_index.pkl') -> None:
        """Save the inverted index to disk using pickle."""
        import pickle
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    def remove_document(self, doc_id: str) -> None:
        """
        Remove a document from the inverted index.
        
        Args:
            doc_id: Document identifier to remove
        """
        for term, docs in list(self._index.items()):
            if doc_id in docs:
                del docs[doc_id]
                self._doc_frequency[term] -= 1
                
                # Clean up if term no longer exists
                if self._doc_frequency[term] == 0:
                    del self._doc_frequency[term]
        
        # Remove document length
        if doc_id in self._doc_lengths:
            del self._doc_lengths[doc_id]
        
        self.N -= 1
    
    def get_candidates(self, terms: List[str]) -> Set[str]:
        """
        Get candidate documents containing any of the given terms.
        
        Args:
            terms: List of query terms
            
        Returns:
            Set of document IDs that contain at least one term
        """
        candidates = set()
        
        for term in terms:
            if term in self._index:
                candidates.update(self._index[term].keys())
        
        return candidates
    
    def get_term_frequency(self, term: str, doc_id: str) -> int:
        """
        Get frequency of a term in a specific document.
        
        Args:
            term: The term to look up
            doc_id: The document ID
            
        Returns:
            Term frequency in the document (0 if not found)
        """
        if term in self._index and doc_id in self._index[term]:
            return self._index[term][doc_id]
        return 0
    
    def get_documents_for_term(self, term: str) -> Dict[str, int]:
        """
        Get all documents containing a term with their frequencies.
        
        Args:
            term: The term to look up
            
        Returns:
            Dictionary mapping doc_id to term frequency
        """
        return self._index.get(term, {}).copy()
    
    def get_postings(self, term: str) -> List[tuple]:
        """
        Get postings for a term as a list of (doc_id, frequency) tuples.
        """
        return list(self.get_documents_for_term(term).items())
    
    def get_document_length(self, doc_id: str) -> int:
        """Get length of a document."""
        return self._doc_lengths.get(doc_id, 0)
    
    def get_average_document_length(self) -> float:
        """Get average document length."""
        if not self._doc_lengths:
            return 0.0
        return sum(self._doc_lengths.values()) / len(self._doc_lengths)
    
    @property
    def doc_frequency(self) -> Dict[str, int]:
        """Get document frequency for all terms."""
        return dict(self._doc_frequency)
    
    @property
    def unique_terms(self) -> int:
        """Get number of unique terms in the index."""
        return len(self._index)
    
    @property
    def total_entries(self) -> int:
        """Get total number of term-document entries."""
        return sum(len(docs) for docs in self._index.values())
    
    @property
    def total_documents(self) -> int:
        """Get total number of documents."""
        return self.N
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the inverted index."""
        return {
            'unique_terms': self.unique_terms,
            'total_entries': self.total_entries,
            'total_documents': self.N,
            'unique_documents': len(self._doc_lengths),
        }
    
    # ========================================================================
    # Magic methods (dunder methods) for better usability
    # ========================================================================
    
    def __len__(self) -> int:
        """
        Return number of unique terms in the index.
        
        This allows: len(index)
        
        Returns:
            Number of unique terms
        """
        return self.unique_terms
    
    def __contains__(self, term: str) -> bool:
        """
        Check if term exists in index.
        
        This allows: term in index
        
        Args:
            term: Term to check
            
        Returns:
            True if term exists, False otherwise
        """
        return term in self._index
    
    def __getitem__(self, term: str) -> Dict[str, int]:
        """
        Get postings for a term using dictionary-style access.
        
        This allows: index[term]
        
        Args:
            term: Term to look up
            
        Returns:
            Dictionary mapping doc_id to term frequency (empty if not found)
        """
        return self._index.get(term, {}).copy()
    
    def __setitem__(self, term: str, postings: Dict[str, int]) -> None:
        """
        Set postings for a term (use with caution).
        
        This allows: index[term] = {...}
        
        Args:
            term: Term to add/update
            postings: Dictionary of doc_id -> term frequency
        """
        # Update document frequencies
        for doc_id, count in postings.items():
            if doc_id not in self._index.get(term, {}):
                self._doc_frequency[term] += 1
        
        self._index[term] = postings
    
    def __delitem__(self, term: str) -> None:
        """
        Delete a term from the index.
        
        This allows: del index[term]
        
        Args:
            term: Term to delete
        """
        if term in self._index:
            del self._doc_frequency[term]
            del self._index[term]
    
    def __iter__(self):
        """
        Iterate over terms in the index.
        
        This allows: for term in index:
        """
        return iter(self._index.keys())
    
    def __repr__(self) -> str:
        """String representation of the inverted index."""
        return f"InvertedIndex(terms={self.unique_terms}, docs={self.N})"
    
    def __str__(self) -> str:
        """User-friendly string representation."""
        return f"InvertedIndex with {self.unique_terms} unique terms across {self.N} documents"