# services/indexing/document_store.py
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any

class DocumentStore:
    """
    Shared DocumentStore component.
    Loads and caches documents to avoid multiple disk reads,
    and provides access to document texts, lengths, and average length.
    """
    _instance = None
    _documents: Dict[str, dict] = {}
    _doc_lengths: Dict[str, int] = {}
    _avg_doc_length: float = 0.0
    _is_loaded = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DocumentStore, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Already initialized via singleton
        pass

    def load(self, file_paths: List[str]) -> int:
        """
        Loads documents from the first available pickle file and caches them.
        """
        if self._is_loaded:
            return len(self._documents)
        
        for path in file_paths:
            if Path(path).exists():
                with open(path, 'rb') as f:
                    docs = pickle.load(f)
                
                print(f"📂 [DocumentStore] Loading {len(docs)} documents from {path}")
                total_len = 0
                for doc in docs:
                    doc_id = str(doc.get('doc_id', ''))
                    if doc_id:
                        text = doc.get('original', doc.get('text', ''))
                        tokens = doc.get('tokens', [])
                        self._documents[doc_id] = {
                            'doc_id': doc_id,
                            'text': text,
                            'tokens': tokens
                        }
                        self._doc_lengths[doc_id] = len(tokens)
                        total_len += len(tokens)
                
                if len(self._documents) > 0:
                    self._avg_doc_length = total_len / len(self._documents)
                
                self._is_loaded = True
                break
        
        return len(self._documents)

    def get_doc(self, doc_id: str) -> Optional[dict]:
        """Returns the document dictionary containing doc_id, text, and tokens."""
        return self._documents.get(str(doc_id))

    def get_length(self, doc_id: str) -> int:
        """Returns the token length of a document."""
        return self._doc_lengths.get(str(doc_id), 0)

    @property
    def avg_doc_length(self) -> float:
        """Returns the average document length across the loaded collection."""
        return self._avg_doc_length

    @property
    def total_docs(self) -> int:
        """Returns the total number of documents in the store."""
        return len(self._documents)

    def get_all_documents(self) -> Dict[str, dict]:
        """Returns all documents."""
        return self._documents