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
    and applies BM25Scorer mathematical ranking.
    """
    def __init__(
        self,
        query_processor: Optional[QueryProcessor] = None,
        document_store: Optional[DocumentStore] = None,
        inverted_index: Optional[InvertedIndex] = None,
        k1: float = 1.2,
        b: float = 0.75
    ):
        self.query_processor = query_processor or QueryProcessor()
        self.document_store = document_store or DocumentStore()
        
        # Load document store cache if it has no documents
        if self.document_store.total_docs == 0:
            doc_paths = [
                'data/processed/processed_docs_200000.pkl',
                'data/processed/processed_docs_5000.pkl',
                'data/processed/processed_docs.pkl',
            ]
            self.document_store.load(doc_paths)
            
        self.inverted_index = inverted_index or self._load_inverted_index()
        
        # Initialize the pure math scorer
        self.scorer = BM25Scorer(k1=k1, b=b)
        
        # Precompute IDF values for faster online scoring
        self.idf_values: Dict[str, float] = {}
        self._precompute_idf()

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

    def _precompute_idf(self) -> None:
        total_docs = self.document_store.total_docs
        if total_docs == 0:
            return
        
        doc_freqs = self.inverted_index.doc_frequency
        for term, df in doc_freqs.items():
            self.idf_values[term] = self.scorer.compute_idf(df, total_docs)

    @property
    def k1(self) -> float:
        return self.scorer.k1

    @k1.setter
    def k1(self, value: float):
        self.scorer.k1 = value

    @property
    def b(self) -> float:
        return self.scorer.b

    @b.setter
    def b(self, value: float):
        self.scorer.b = value

    def get_stats(self) -> Dict[str, Any]:
        """Returns index and store stats for the UI."""
        return {
            'unique_terms': len(self.inverted_index),
            'total_documents': self.document_store.total_docs,
            'avg_doc_length': self.document_store.avg_doc_length
        }

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Executes query retrieval and BM25 ranking.
        """
        # 1. Query processing
        query_tokens = self.query_processor.process_query(query)
        if not query_tokens:
            return []
        
        # 2. Candidate retrieval
        candidates = self.inverted_index.get_candidates(query_tokens)
        
        # 3. BM25 Scoring
        scores = []
        avg_doc_len = self.document_store.avg_doc_length
        total_docs = self.document_store.total_docs
        
        for doc_id in candidates:
            total_score = 0.0
            doc_len = self.document_store.get_length(doc_id)
            
            for term in query_tokens:
                # get term frequency in this document from inverted index
                tf = self.inverted_index.get_term_frequency(term, doc_id)
                if tf > 0:
                    idf = self.idf_values.get(term)
                    if idf is None:
                        # compute on the fly if query term not precomputed
                        df = self.inverted_index.doc_frequency.get(term, 0)
                        idf = self.scorer.compute_idf(df, total_docs)
                    
                    total_score += self.scorer.score_term(tf, doc_len, avg_doc_len, idf)
            
            if total_score > 0:
                scores.append((doc_id, total_score))
                
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
                'method': f'BM25 (k1={self.scorer.k1}, b={self.scorer.b})'
            })
        return results