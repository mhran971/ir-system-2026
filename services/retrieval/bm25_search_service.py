# services/retrieval/bm25_search_service.py
import os
import pickle
from typing import List, Dict, Any, Optional
from services.query_processing.query_processor import QueryProcessor
from services.indexing.document_store import DocumentStore
from services.indexing.inverted_index import InvertedIndex
from services.ranking.bm25_scorer import BM25Scorer

class BM25SearchService:
    """
    Service layer orchestrator for the BM25 retrieval model.
    Coordinates QueryProcessor, InvertedIndex, and DocumentStore,
    and applies BM25 scoring directly using the inverted index postings.
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
        
        # Load document store cache (metadata only)
        if self.document_store.total_docs == 0:
            doc_paths = [
                'data/processed/processed_docs.pkl',
            ]
            self.document_store.load(doc_paths)
            
        self.inverted_index = inverted_index or self._load_inverted_index()
        
        self._k1 = k1
        self._b = b
        self.scorer = BM25Scorer(k1=k1, b=b)

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
        
        print("⚠️ [BM25SearchService] No inverted index found on disk. Building from store...")
        index = InvertedIndex()
        # Fallback to build new index from document store (this uses lazy docs but will require texts)
        # Note: In production this path should not be hit.
        for doc_id in self.document_store._doc_lengths.keys():
            doc = self.document_store.get_doc(doc_id)
            if doc:
                index.add_document(doc_id, doc.get('tokens', []))
        return index

    def _rebuild_bm25(self):
        """Rebuild BM25 scorer with current parameters (instant)."""
        self.scorer = BM25Scorer(k1=self._k1, b=self._b)

    @property
    def k1(self) -> float:
        return self._k1

    @k1.setter
    def k1(self, value: float):
        if value <= 0:
            raise ValueError("k1 must be > 0")
        self._k1 = value
        self._rebuild_bm25()

    @property
    def b(self) -> float:
        return self._b

    @b.setter
    def b(self, value: float):
        if not 0 <= value <= 1:
            raise ValueError("b must be between 0 and 1")
        self._b = value
        self._rebuild_bm25()

    def get_stats(self) -> Dict[str, Any]:
        """Returns index and store stats for the UI."""
        return {
            'unique_terms': len(self.inverted_index),
            'total_documents': self.inverted_index.total_documents,
            'avg_doc_length': self.inverted_index.get_average_document_length(),
            'k1': self._k1,
            'b': self._b
        }

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Executes query retrieval and BM25 ranking directly using the inverted index.
        """
        if not query or not query.strip():
            return []

        query_tokens = self.query_processor.process_query(query)
        if not query_tokens:
            return []
            
        scores = {}
        total_docs = self.inverted_index.total_documents
        avg_doc_len = self.inverted_index.get_average_document_length()
        
        # Calculate scores for candidate documents containing at least one query term
        for token in query_tokens:
            postings = self.inverted_index.get_documents_for_term(token)
            if not postings:
                continue
            
            df = len(postings)
            idf = self.scorer.compute_idf(df, total_docs)
            if idf <= 0:
                continue
                
            for doc_id, tf in postings.items():
                doc_len = self.inverted_index.get_document_length(doc_id)
                score = self.scorer.score_term(tf, doc_len, avg_doc_len, idf)
                scores[doc_id] = scores.get(doc_id, 0.0) + score
                
        if not scores:
            return []
            
        # Sort and return top_k
        top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
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
                'method': f'BM25 (k1={self._k1}, b={self._b})'
            })
        return results

    def get_term_score(self, doc_id: str, term: str) -> Dict[str, float]:
        """
        Get detailed term score for a specific document and term.
        """
        tf = self.inverted_index.get_term_frequency(term, doc_id)
        if tf == 0:
            return {'tf': 0.0, 'idf': 0.0, 'score': 0.0}
            
        doc_len = self.inverted_index.get_document_length(doc_id)
        avg_doc_len = self.inverted_index.get_average_document_length()
        
        df = len(self.inverted_index.get_documents_for_term(term))
        idf = self.scorer.compute_idf(df, self.inverted_index.total_documents)
        score = self.scorer.score_term(tf, doc_len, avg_doc_len, idf)
        
        return {
            'tf': tf / doc_len if doc_len > 0 else 0.0,
            'idf': idf,
            'score': score
        }