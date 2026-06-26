# services/retrieval/bm25_search_service.py
import os
import pickle
from typing import List, Dict, Any, Optional
from services.query_processing.query_processor import QueryProcessor
from services.indexing.document_store import DocumentStore
from services.indexing.inverted_index import InvertedIndex
from rank_bm25 import BM25Okapi  # ✅ مكتبة rank_bm25


class BM25SearchService:
    """
    Service layer orchestrator for the BM25 retrieval model.
    Uses rank_bm25 library for fast BM25 scoring.
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
        
        # ✅ استخدام rank_bm25
        self.bm25 = None
        self.doc_ids = []
        self._load_or_build_bm25()

    def _load_inverted_index(self) -> InvertedIndex:
        """Load inverted index from disk."""
        index_paths = [
            'data/index/inverted_index.pkl',
            'data/index/bm25_index.pkl',
            'data/index/index.pkl'
        ]
        for path in index_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        data = pickle.load(f)
                    if isinstance(data, dict) and 'inverted_index' in data:
                        from collections import defaultdict
                        index = InvertedIndex()
                        index._index = defaultdict(dict, {k: dict(v) for k, v in data['inverted_index'].items()})
                        index._doc_frequency = defaultdict(int, data.get('doc_frequency', {}))
                        index._doc_lengths = dict(data.get('doc_lengths', {}))
                        index.N = data.get('total_docs', len(index._doc_lengths))
                        print(f"✅ [BM25SearchService] Loaded InvertedIndex from {path}")
                        return index
                    elif isinstance(data, InvertedIndex):
                        print(f"✅ [BM25SearchService] Loaded InvertedIndex from {path}")
                        return data
                except Exception as e:
                    print(f"⚠️ [BM25SearchService] Error loading index from {path}: {e}")
        
        print("⚠️ [BM25SearchService] No inverted index found. Building from store...")
        index = InvertedIndex()
        for doc_id in self.document_store._doc_lengths.keys():
            doc = self.document_store.get_doc(doc_id)
            if doc:
                index.add_document(doc_id, doc.get('tokens', []))
        return index

    def _load_or_build_bm25(self):
        """Load precomputed BM25 model or build on-demand."""
        # محاولة تحميل النموذج المحفوظ
        bm25_model_path = 'data/index/bm25_model.pkl'
        
        if os.path.exists(bm25_model_path):
            try:
                with open(bm25_model_path, 'rb') as f:
                    data = pickle.load(f)
                self.bm25 = data['bm25']
                self.doc_ids = data['doc_ids']
                # تحديث المعاملات
                self.bm25.k1 = self._k1
                self.bm25.b = self._b
                print(f"✅ [BM25SearchService] Loaded precomputed BM25 model from {bm25_model_path}")
                return
            except Exception as e:
                print(f"⚠️ [BM25SearchService] Error loading BM25 model: {e}. Building from store...")
        
        # بناء النموذج من الصفر
        print("⏳ [BM25SearchService] Building BM25 model from document store...")
        self.doc_ids = list(self.document_store.get_all_documents().keys())
        tokenized_corpus = []
        for doc_id in self.doc_ids:
            doc = self.document_store.get_doc(doc_id)
            tokenized_corpus.append(doc.get('tokens', []) if doc else [])
        
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self._k1, b=self._b)
        print(f"✅ [BM25SearchService] BM25 model built with {len(self.doc_ids)} documents")

    @property
    def k1(self) -> float:
        return self._k1

    @k1.setter
    def k1(self, value: float):
        if value <= 0:
            raise ValueError("k1 must be > 0")
        self._k1 = value
        if self.bm25:
            self.bm25.k1 = value

    @property
    def b(self) -> float:
        return self._b

    @b.setter
    def b(self, value: float):
        if not 0 <= value <= 1:
            raise ValueError("b must be between 0 and 1")
        self._b = value
        if self.bm25:
            self.bm25.b = value

    def get_stats(self) -> Dict[str, Any]:
        """Returns index and store stats for the UI."""
        return {
            'unique_terms': len(self.inverted_index),
            'total_documents': len(self.doc_ids) if self.doc_ids else self.inverted_index.total_documents,
            'avg_doc_length': self.bm25.avgdl if self.bm25 else self.inverted_index.get_average_document_length(),
            'k1': self._k1,
            'b': self._b
        }

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Executes query retrieval and BM25 ranking using rank_bm25 library.
        """
        if not query or not query.strip():
            return []

        query_tokens = self.query_processor.process_query(query)
        if not query_tokens or self.bm25 is None or not self.doc_ids:
            return []
        
        # ✅ حساب الدرجات باستخدام rank_bm25
        doc_scores = self.bm25.get_scores(query_tokens)
        
        # ربط الدرجات بمعرفات الوثائق
        scores = []
        for doc_id, score in zip(self.doc_ids, doc_scores):
            if score > 0:
                scores.append((doc_id, float(score)))
        
        if not scores:
            return []
        
        # ترتيب النتائج
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
                'method': f'BM25 (k1={self._k1}, b={self._b})'
            })
        return results

    def get_term_score(self, doc_id: str, term: str) -> Dict[str, float]:
        """
        Get detailed term score for a specific document and term.
        """
        if self.bm25 is None:
            return {'tf': 0.0, 'idf': 0.0, 'score': 0.0}
        
        # الحصول على TF من الفهرس
        tf = self.inverted_index.get_term_frequency(term, doc_id)
        if tf == 0:
            return {'tf': 0.0, 'idf': 0.0, 'score': 0.0}
        
        doc_len = self.inverted_index.get_document_length(doc_id)
        avg_doc_len = self.bm25.avgdl
        
        # الحصول على IDF من نموذج BM25
        idf = self.bm25.idf.get(term, 0.0)
        
        # حساب مساهمة المصطلح باستخدام صيغة BM25
        k1 = self._k1
        b = self._b
        denominator = tf + k1 * (1 - b + b * doc_len / avg_doc_len)
        score = idf * (tf * (k1 + 1)) / denominator if denominator > 0 else 0.0
        
        return {
            'tf': tf / doc_len if doc_len > 0 else 0.0,
            'idf': idf,
            'score': score
        }