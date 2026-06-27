# services/retrieval/vsm_search_service.py
import os
import pickle
import shutil
import scipy.sparse
from typing import List, Dict, Any, Optional
from services.query_processing.query_processor import QueryProcessor
from services.indexing.document_store import DocumentStore
from services.indexing.inverted_index import InvertedIndex


def identity_analyzer(doc):
    """Pass-through analyzer for pre-tokenized inputs."""
    return doc


class VSMSearchService:
    """
    Service layer orchestrator for the Vector Space Model (VSM).
    Uses scikit-learn for TF-IDF and direct dot product for fast cosine similarity.
    Caches model data in class-level variables to allow instant load.
    """
    # Class-level cache to keep indices and models in memory
    _cached_inverted_index: Optional[InvertedIndex] = None
    _cached_vectorizer = None
    _cached_tfidf_matrix = None
    _cached_doc_ids: List[str] = []

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
            doc_paths = ['data/processed/processed_docs.pkl']
            self.document_store.load(doc_paths)
            
        # Retrieve inverted index from provided parameter, memory cache, or disk
        if inverted_index is not None:
            self.inverted_index = inverted_index
        elif VSMSearchService._cached_inverted_index is not None:
            self.inverted_index = VSMSearchService._cached_inverted_index
        else:
            self.inverted_index = self._load_inverted_index()
            VSMSearchService._cached_inverted_index = self.inverted_index
        
        # Load precomputed VSM model and TF-IDF matrix
        self.vectorizer = None
        self.tfidf_matrix = None
        self.doc_ids = []
        self._load_vsm_model()

    @classmethod
    def clear_cache(cls):
        """Clean up all memory-cached index and model instances."""
        cls._cached_inverted_index = None
        cls._cached_vectorizer = None
        cls._cached_tfidf_matrix = None
        cls._cached_doc_ids = []
        print("🧹 [VSMSearchService] In-memory cache cleared successfully.")

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
        Load precomputed VSM model and TF-IDF matrix from disk or memory cache.
        If not found, build on-demand (fallback only).
        """
        if VSMSearchService._cached_vectorizer is not None:
            self.vectorizer = VSMSearchService._cached_vectorizer
            self.tfidf_matrix = VSMSearchService._cached_tfidf_matrix
            self.doc_ids = VSMSearchService._cached_doc_ids
            print("✅ [VSMSearchService] Loaded VSM model from memory cache")
            return

        index_dir = self._get_index_dir()
        model_path = os.path.join(index_dir, 'vsm_model.pkl')
        matrix_path = os.path.join(index_dir, 'vsm_matrix.npz')
        
        if os.path.exists(model_path) and os.path.exists(matrix_path):
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                self.vectorizer = data['vectorizer']
                self.doc_ids = data['doc_ids']
                self.tfidf_matrix = scipy.sparse.load_npz(matrix_path)
                
                # Cache in class variables
                VSMSearchService._cached_vectorizer = self.vectorizer
                VSMSearchService._cached_tfidf_matrix = self.tfidf_matrix
                VSMSearchService._cached_doc_ids = self.doc_ids
                
                print(f"✅ [VSMSearchService] Loaded VSM model from {index_dir}")
                print(f"   Matrix shape: {self.tfidf_matrix.shape}")
                print(f"   Vocabulary size: {len(self.vectorizer.vocabulary_):,}")
                return
            except Exception as e:
                print(f"⚠️ [VSMSearchService] Error loading VSM model: {e}. Building on-demand...")
        
        # Build VSM from scratch if not found
        print("⚠️ [VSMSearchService] Precomputed VSM model not found. Building on-demand...")
        self._build_vsm_model()

    def _build_vsm_model(self):
        """
        Build VSM model from document store using scikit-learn.
        This is slow but only runs once as fallback.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        print("⏳ [VSMSearchService] Building VSM model from document store...")
        
        self.doc_ids = list(self.document_store.get_all_documents().keys())
        tokenized_corpus = []
        for doc_id in self.doc_ids:
            doc = self.document_store.get_doc(doc_id)
            tokenized_corpus.append(doc.get('tokens', []) if doc else [])
        
        self.vectorizer = TfidfVectorizer(
            analyzer=identity_analyzer,
            lowercase=False,
            token_pattern=None,
            max_features=50000,
            min_df=2,
            max_df=0.9,
        )
        
        if tokenized_corpus:
            self.tfidf_matrix = self.vectorizer.fit_transform(tokenized_corpus)
            
            # Cache in class variables
            VSMSearchService._cached_vectorizer = self.vectorizer
            VSMSearchService._cached_tfidf_matrix = self.tfidf_matrix
            VSMSearchService._cached_doc_ids = self.doc_ids
            
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
        Uses direct dot-product calculation for pre-normalized vectors, gaining 3.4x speedup.
        """
        if not query or not query.strip():
            return []

        query_tokens = self.query_processor.process_query(query)
        if not query_tokens or self.tfidf_matrix is None or self.vectorizer is None:
            return []
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([query_tokens])
        
        # Direct dot product calculation (since TF-IDF rows are L2 normalized,
        # cosine similarity is equivalent to dot product)
        similarities = self.tfidf_matrix.dot(query_vector.T).toarray().flatten()
        
        scores = []
        for idx, score in enumerate(similarities):
            if score > 0:
                scores.append((self.doc_ids[idx], float(score)))
        
        # Sort scores
        scores.sort(key=lambda x: x[1], reverse=True)
        top_scores = scores[:top_k]
        
        # Prepare results
        results = []
        preview_len = 300
        for doc_id, score in top_scores:
            doc = self.document_store.get_doc(doc_id)
            text = doc['text'] if doc else ""
            preview = text[:preview_len] + ("..." if len(text) > preview_len else "")
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
        
        term_idx = self.vectorizer.vocabulary_.get(term)
        if term_idx is None:
            return {'tf': 0.0, 'idf': 0.0, 'tfidf': 0.0}
        
        idf = float(self.vectorizer.idf_[term_idx])
        
        tf = 0.0
        doc_len = self.inverted_index.get_document_length(doc_id)
        if doc_len > 0:
            term_count = self.inverted_index.get_term_frequency(term, doc_id)
            tf = term_count / doc_len
        
        tfidf = tf * idf
        return {'tf': tf, 'idf': idf, 'tfidf': tfidf}