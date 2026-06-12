# services/ranking/document_store.py
"""
Document storage and loading management.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional

from services.preprocessing.preprocessor import TextPreprocessor
from .models import Document


class DocumentStore:
    """
    Manages document storage, loading, and retrieval.
    """
    
    def __init__(self, preprocessor: Optional[TextPreprocessor] = None):
        """
        Initialize document store.
        
        Args:
            preprocessor: Text preprocessor for cleaning documents
        """
        self._documents: Dict[str, Document] = {}
        self._preprocessor = preprocessor or TextPreprocessor()
    
    def add_document(self, doc_id: str, text: str, tokens: Optional[List[str]] = None) -> None:
        """
        Add a document to the store.
        
        Args:
            doc_id: Document identifier
            text: Original document text
            tokens: Preprocessed tokens (optional, will be generated if not provided)
        """
        if tokens is None and text:
            tokens = self._preprocessor.process(text)
        
        self._documents[doc_id] = Document(
            doc_id=doc_id,
            text=text,
            tokens=tokens or []
        )
    
    def add_documents_from_list(self, documents: List[dict]) -> None:
        """
        Add multiple documents from a list of dictionaries.
        
        Each document should have at least 'doc_id' and either 'original'/'text' fields.
        May also have 'tokens' field for preprocessed tokens.
        
        Args:
            documents: List of document dictionaries
        """
        for doc in documents:
            doc_id = str(doc['doc_id'])
            text = doc.get('original', doc.get('text', ''))
            tokens = doc.get('tokens', [])
            
            self.add_document(doc_id, text, tokens)
    
    def load_from_pickle(self, file_paths: List[str]) -> int:
        """
        Load documents from pickle files.
        
        Args:
            file_paths: List of file paths to try (uses first existing file)
            
        Returns:
            Number of documents loaded
        """
        for file_path in file_paths:
            if Path(file_path).exists():
                return self._load_single_file(file_path)
        
        return 0
    
    def _load_single_file(self, file_path: str) -> int:
        """
        Load documents from a single pickle file.
        
        Args:
            file_path: Path to pickle file
            
        Returns:
            Number of documents loaded
        """
        with open(file_path, 'rb') as f:
            docs = pickle.load(f)
        
        print(f"📂 Loading {len(docs)} documents from {file_path}")
        self.add_documents_from_list(docs)
        
        return len(docs)
    
    def get(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        return self._documents.get(doc_id)
    
    def get_text(self, doc_id: str) -> str:
        """Get document text by ID."""
        doc = self.get(doc_id)
        return doc.text if doc else ""
    
    def get_tokens(self, doc_id: str) -> List[str]:
        """Get document tokens by ID."""
        doc = self.get(doc_id)
        return doc.tokens if doc else []
    
    def get_all_documents(self) -> Dict[str, Document]:
        """Get all documents."""
        return self._documents.copy()
    
    def get_all_ids(self) -> List[str]:
        """Get all document IDs."""
        return list(self._documents.keys())
    
    @property
    def total_docs(self) -> int:
        """Get total number of documents."""
        return len(self._documents)
    
    def __contains__(self, doc_id: str) -> bool:
        """Check if document exists."""
        return doc_id in self._documents
    
    def __len__(self) -> int:
        """Get total number of documents."""
        return self.total_docs