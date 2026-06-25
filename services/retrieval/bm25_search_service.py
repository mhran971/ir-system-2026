# services/retrieval/bm25_search_service.py
import os
import pickle
from typing import List, Dict, Any, Optional
from services.query_processing.query_processor import QueryProcessor
from services.indexing.document_store import DocumentStore
from services.indexing.inverted_index import InvertedIndex
from rank_bm25 import BM25Okapi

class BM25ScorerCompat:
    """
    Compatibility layer for BM25Scorer.
    Allows UI elements to query individual term scores without using the manual scorer class directly.
    """
    def __init__(self, parent_service):
        self.parent = parent_service

    @property
    def k1(self) -> float:
        return self.parent.k1

    @k1.setter
    def k1(self, value: float):
        self.parent.k1 = value

    @property
    def b(self) -> float:
        return self.parent.b

    @b.setter
    def b(self, value: float):
        self.parent.b = value

    def compute_idf(self, df: int, total_docs: int) -> float:
        import math
        numerator = total_docs - df + 0.5
        denominator = df + 0.5
        return max(math.log(numerator / denominator + 1e-10), 0.0)

    def score_term(self, tf: int, doc_len: int, avg_doc_len: float, idf: float) -> float:
        if tf <= 0 or idf <= 0:
            return 0.0
        length_norm = (1 - self.b) + self.b * (doc_len / avg_doc_len)
        tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * length_norm)
        return idf * tf_component


class BM25SearchService:
    """
    Service layer orchestrator for the BM25 retrieval model.
    Coordinates QueryProcessor, InvertedIndex, and DocumentStore,
    and applies rank_bm25 library for ranking.
    """
    def __init__(
        self,
        query_processor: Optional[QueryProcessor] = None,
        document_store: Optional[DocumentStore] = None,
        inverted_index: Optional[InvertedIndex] = None,
        k1: float = 1.5,
        b: float = 0.75
    ):
        self.query_processor = query_processor or QueryProcessor()
        self.document_store = document_store or DocumentStore()
        
        # Load document store cache if it has no documents
        if self.document_store.total_docs == 0:
            doc_paths = [
                'data/processed/processed_docs.pkl',
            ]
            self.document_store.load(doc_paths)
            
        self.inverted_index = inverted_index or self._load_inverted_index()
        
        # Get document IDs and tokenized corpus
        self.doc_ids = list(self.document_store.get_all_documents().keys())
        tokenized_corpus = [self.document_store.get_doc(doc_id)['tokens'] for doc_id in self.doc_ids]
        
        # Initialize BM25Okapi
        if self.doc_ids:
            self.bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
        else:
            self.bm25 = None
            
        # Precompute IDF values for faster online scoring / compatibility
        self.idf_values: Dict[str, float] = self.bm25.idf if self.bm25 else {}
        
        # Initialize compatibility scorer to keep UI working
        self.scorer = BM25ScorerCompat(self)

    def _load_inverted_index(self) -> InvertedIndex:
        index_paths = [
            'data/index/inverted_index.pkl',
            'data/index/index.pkl'
        ]
        for path in index_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        index = pickle.load(f)
                    print(f"✅ [BM25SearchService] Loaded InvertedIndex from {path}")
                    return index
                except Exception as e:
                    print(f"⚠️ [BM25SearchService] Error loading index from {path}: {e}")
        
        # Fallback to build new index from document store if not found
        print("⚠️ [BM25SearchService] No inverted index found on disk. Building from store...")
        index = InvertedIndex()
        for doc_id, doc in self.document_store.get_all_documents().items():
            index.add_document(doc_id, doc['tokens'])
        return index

    @property
    def k1(self) -> float:
        return self.bm25.k1 if self.bm25 else 1.5

    @k1.setter
    def k1(self, value: float):
        if self.bm25:
            self.bm25.k1 = value

    @property
    def b(self) -> float:
        return self.bm25.b if self.bm25 else 0.75

    @b.setter
    def b(self, value: float):
        if self.bm25:
            self.bm25.b = value

    def get_stats(self) -> Dict[str, Any]:
        """Returns index and store stats for the UI."""
        return {
            'unique_terms': len(self.inverted_index),
            'total_documents': self.document_store.total_docs,
            'avg_doc_length': self.document_store.avg_doc_length
        }

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Executes query retrieval and BM25 ranking using rank_bm25 library.
        """
        # Handle empty/whitespace-only queries
        if not query or not query.strip():
            return []

        # 1. Query processing
        query_tokens = self.query_processor.process_query(query)
        if not query_tokens or self.bm25 is None:
            return []
        
        # 2. Score using rank_bm25
        doc_scores = self.bm25.get_scores(query_tokens)
        
        # 3. Associate scores with doc_ids
        scores = []
        for doc_id, score in zip(self.doc_ids, doc_scores):
            if score > 0:
                scores.append((doc_id, float(score)))
                
        # 4. Sort and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        top_scores = scores[:top_k]
        
        results = []
        for doc_id, score in top_scores:
            doc = self.document_store.get_doc(doc_id)
            text = doc['text'] if doc else ""
            preview = text[:300] + ("..." if len(text) > 300 else "")
            results.append({
                'doc_id': doc_id,
                'score': round(score, 6),
                'text': preview,
                'full_text': text,
                'method': f'BM25 (k1={self.k1}, b={self.b})'
            })
        return results