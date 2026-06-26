# services/retrieval/vsm_search_service.py
import os
import pickle
import shutil
import scipy.sparse
from typing import List, Dict, Any, Optional
from services.query_processing.query_processor import QueryProcessor
from services.indexing.document_store import DocumentStore
from services.indexing.inverted_index import InvertedIndex
from sklearn.metrics.pairwise import cosine_similarity  # ✅ sklearn
from sklearn.feature_extraction.text import TfidfVectorizer  # ✅ sklearn


def identity_analyzer(doc):
    """Pass-through analyzer for pre-tokenized inputs."""
    return doc


class VSMSearchService:
    """
    Service layer orchestrator for the Vector Space Model (VSM).
    Uses scikit-learn for TF-IDF and Cosine Similarity.
    """
    def __init__(
        self,
        query_processor: Optional[QueryProcessor] = None,
        document_store: Optional[DocumentStore] = None,
        inverted_index: Optional[InvertedIndex] = None
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
        
        # Load precomputed VSM model and TF-IDF matrix
        self.vectorizer = None
        self.tfidf_matrix = None
        self.doc_ids = []
        self._load_vsm_model()

    def _get_index_dir(self) -> str:
        """Get index directory with fallback to C: drive if Y: is full."""
        default_dir = 'data/index'
        try:
            abs_path = os.path.abspath(default_dir)
            drive = os.path.splitdrive(abs_path)[0]
            usage = shutil.disk_usage(drive)
            if usage.free < 1.5 * 1024 * 1024 * 1024:
                fallback_dir = os.path.expanduser('~/.ir_system_cache')
                if os.path.exists(fallback_dir):
                    return fallback_dir
        except Exception:
            pass
        return default_dir

    def _load_inverted_index(self) -> InvertedIndex:
        """Load inverted index from disk."""
        index_paths = [
            'data/index/inverted_index.pkl',
            'data/index/bm25_index.pkl',
            'data/index/index.pkl'
        ]
        index_dir = self._get_index_dir()
        index_paths.insert(0, os.path.join(index_dir, 'inverted_index.pkl'))
        
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
                        print(f"✅ [VSMSearchService] Loaded InvertedIndex from {path}")
                        return index
                    elif isinstance(data, InvertedIndex):
                        print(f"✅ [VSMSearchService] Loaded InvertedIndex from {path}")
                        return data
                except Exception as e:
                    print(f"⚠️ [VSMSearchService] Error loading index from {path}: {e}")
        
        print("⚠️ [VSMSearchService] No inverted index found. Building from store...")
        index = InvertedIndex()
        for doc_id in self.document_store._doc_lengths.keys():
            doc = self.document_store.get_doc(doc_id)
            if doc:
                index.add_document(doc_id, doc.get('tokens', []))
        return index

    def _load_vsm_model(self):
        """
        Load precomputed VSM model and TF-IDF matrix from disk.
        If not found, build on-demand (slow, fallback only).
        """
        index_dir = self._get_index_dir()
        model_path = os.path.join(index_dir, 'vsm_model.pkl')
        matrix_path = os.path.join(index_dir, 'vsm_matrix.npz')
        
        # ✅ محاولة تحميل النموذج المحفوظ
        if os.path.exists(model_path) and os.path.exists(matrix_path):
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                self.vectorizer = data['vectorizer']
                self.doc_ids = data['doc_ids']
                self.tfidf_matrix = scipy.sparse.load_npz(matrix_path)
                print(f"✅ [VSMSearchService] Loaded precomputed VSM model from {index_dir}")
                print(f"   Matrix shape: {self.tfidf_matrix.shape}")
                print(f"   Vocabulary size: {len(self.vectorizer.vocabulary_):,}")
                return
            except Exception as e:
                print(f"⚠️ [VSMSearchService] Error loading VSM model: {e}. Building on-demand...")
        
        # ✅ بناء النموذج من الصفر (مرة واحدة فقط، لكنه بطيء)
        print("⚠️ [VSMSearchService] Precomputed VSM model not found. Building on-demand...")
        self._build_vsm_model()

    def _build_vsm_model(self):
        """
        Build VSM model from document store using scikit-learn.
        This is slow but only runs once.
        """
        print("⏳ [VSMSearchService] Building VSM model from document store...")
        
        # جمع جميع الوثائق
        self.doc_ids = list(self.document_store.get_all_documents().keys())
        tokenized_corpus = []
        for doc_id in self.doc_ids:
            doc = self.document_store.get_doc(doc_id)
            tokenized_corpus.append(doc.get('tokens', []) if doc else [])
        
        # ✅ استخدام TfidfVectorizer من sklearn
        self.vectorizer = TfidfVectorizer(
            analyzer=identity_analyzer,
            lowercase=False,
            token_pattern=None,
            max_features=50000,  # حد عدد المصطلحات لتقليل الذاكرة
            min_df=2,  # تجاهل المصطلحات النادرة جداً
            max_df=0.9,  # تجاهل المصطلحات الشائعة جداً
        )
        
        if tokenized_corpus:
            self.tfidf_matrix = self.vectorizer.fit_transform(tokenized_corpus)
            print(f"✅ [VSMSearchService] VSM model built successfully!")
            print(f"   Matrix shape: {self.tfidf_matrix.shape}")
            print(f"   Vocabulary size: {len(self.vectorizer.vocabulary_):,}")
        else:
            self.tfidf_matrix = None
            print("❌ [VSMSearchService] No documents to build VSM model.")

    @property
    def total_docs(self) -> int:
        return len(self.doc_ids) if self.doc_ids else self.inverted_index.total_documents

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Executes query retrieval and VSM Cosine Similarity ranking using scikit-learn.
        """
        if not query or not query.strip():
            return []

        query_tokens = self.query_processor.process_query(query)
        if not query_tokens or self.tfidf_matrix is None or self.vectorizer is None:
            return []
        
        # ✅ تحويل الاستعلام إلى متجه باستخدام sklearn
        query_vector = self.vectorizer.transform([query_tokens])
        
        # ✅ حساب Cosine Similarity باستخدام sklearn
        similarities = cosine_similarity(self.tfidf_matrix, query_vector).flatten()
        
        # ربط الدرجات بمعرفات الوثائق
        scores = []
        for idx, score in enumerate(similarities):
            if score > 0:
                scores.append((self.doc_ids[idx], float(score)))
        
        # ترتيب النتائج
        scores.sort(key=lambda x: x[1], reverse=True)
        top_scores = scores[:top_k]
        
        # تجهيز النتائج
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
                'method': 'VSM TF-IDF (Cosine Similarity)'
            })
        return results

    def get_term_details(self, doc_id: str, term: str) -> Dict[str, float]:
        """
        Get detailed TF, IDF, and TF-IDF values for a term in a document.
        Uses scikit-learn's TfidfVectorizer.
        """
        if self.vectorizer is None:
            return {'tf': 0.0, 'idf': 0.0, 'tfidf': 0.0}
        
        # الحصول على معرف المصطلح من المفردات
        term_idx = self.vectorizer.vocabulary_.get(term)
        if term_idx is None:
            return {'tf': 0.0, 'idf': 0.0, 'tfidf': 0.0}
        
        # ✅ الحصول على IDF من sklearn
        idf = float(self.vectorizer.idf_[term_idx])
        
        # حساب TF من الفهرس المقلوب
        tf = 0.0
        doc_len = self.inverted_index.get_document_length(doc_id)
        if doc_len > 0:
            term_count = self.inverted_index.get_term_frequency(term, doc_id)
            tf = term_count / doc_len
        
        tfidf = tf * idf
        return {'tf': tf, 'idf': idf, 'tfidf': tfidf}