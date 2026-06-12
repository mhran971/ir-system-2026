# services/ranking/models.py
"""
Data models for VSM TF-IDF system.
"""

from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class Document:
    """Represents a document with its content and processed tokens."""
    doc_id: str
    text: str
    tokens: List[str]
    
    @property
    def length(self) -> int:
        """Get document length in tokens."""
        return len(self.tokens)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'doc_id': self.doc_id,
            'text': self.text,
            'tokens': self.tokens,
            'length': self.length
        }


@dataclass
class SearchResult:
    """Represents a search result."""
    doc_id: str
    score: float
    text: str
    full_text: str
    method: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            'doc_id': self.doc_id,
            'score': self.score,
            'text': self.text,
            'full_text': self.full_text,
            'method': self.method
        }


@dataclass
class TermDetails:
    """Detailed information about a term in a document."""
    tf: float = 0.0
    idf: float = 0.0
    tfidf: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {'tf': self.tf, 'idf': self.idf, 'tfidf': self.tfidf}