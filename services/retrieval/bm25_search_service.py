import os
import pickle
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi  #   python -m pip install rank-bm25 
from services.query_processing.query_processor import QueryProcessor
from services.indexing.document_store import DocumentStore
from services.indexing.inverted_index import InvertedIndex


class BM25SearchService:
    """
    BM25 retrieval using the rank_bm25 library (BM25Okapi).
    Replaces the manual BM25Scorer implementation.
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

        if self.document_store.total_docs == 0:
            doc_paths = [
                'data/processed/processed_docs_200000.pkl',
                'data/processed/processed_docs_5000.pkl',
                'data/processed/processed_docs.pkl',
            ]
            self.document_store.load(doc_paths)

        self.inverted_index = inverted_index or self._load_inverted_index()

        self._k1 = k1
        self._b = b

        # Build BM25 index using the library
        self._build_bm25()

    def _load_inverted_index(self) -> InvertedIndex:
        for path in ['data/index/inverted_index.pkl', 'data/index/index.pkl']:
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    print(f"⚠️ [BM25SearchService] {e}")
        index = InvertedIndex()
        for doc_id, doc in self.document_store.get_all_documents().items():
            index.add_document(doc_id, doc['tokens'])
        return index

    def _build_bm25(self) -> None:
        """Build BM25Okapi index from document tokens."""
        all_docs = self.document_store.get_all_documents()
        # Preserve order: doc_ids[i] corresponds to corpus[i]
        self.doc_ids = list(all_docs.keys())
        corpus = [all_docs[doc_id]['tokens'] for doc_id in self.doc_ids]

        # ✅ BM25Okapi من مكتبة rank_bm25
        self.bm25 = BM25Okapi(corpus, k1=self._k1, b=self._b)
        print(f"✅ [BM25SearchService] BM25Okapi built — "
              f"{len(self.doc_ids)} docs, k1={self._k1}, b={self._b}")

    # ── k1 / b properties (rebuild index on change for UI sliders) ──────
    @property
    def k1(self) -> float:
        return self._k1

    @k1.setter
    def k1(self, value: float):
        self._k1 = value
        self._build_bm25()

    @property
    def b(self) -> float:
        return self._b

    @b.setter
    def b(self, value: float):
        self._b = value
        self._build_bm25()

    def get_stats(self) -> Dict[str, Any]:
        return {
            'unique_terms': len(self.inverted_index),
            'total_documents': self.document_store.total_docs,
            'avg_doc_length': self.document_store.avg_doc_length,
        }

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []

        # 1. Preprocess query (same pipeline as documents)
        query_tokens = self.query_processor.process_query(query)
        if not query_tokens:
            return []

        # 2. ✅ Score all docs using rank_bm25
        scores = self.bm25.get_scores(query_tokens)  # numpy array

        # 3. Pair with doc_ids and take top-k
        scored = sorted(
            zip(self.doc_ids, scores),
            key=lambda x: x[1], reverse=True
        )

        results = []
        for doc_id, score in scored[:top_k]:
            if score <= 0:
                continue
            doc = self.document_store.get_doc(doc_id)
            text = doc['text'] if doc else ""
            results.append({
                'doc_id': doc_id,
                'score': round(float(score), 6),
                'text': text[:300] + ('...' if len(text) > 300 else ''),
                'full_text': text,
                'method': f'BM25 — rank_bm25.BM25Okapi (k1={self._k1}, b={self._b})',
            })
        return results