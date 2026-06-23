import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer       
from sklearn.metrics.pairwise import cosine_similarity         # python -m pip install scikit-learn    
from services.query_processing.query_processor import QueryProcessor
from services.indexing.document_store import DocumentStore
from services.indexing.inverted_index import InvertedIndex


class VSMSearchService:
    """
    TF-IDF / VSM retrieval using sklearn's TfidfVectorizer.
    Replaces the manual VSMScorer implementation.
    """
    def __init__(
        self,
        query_processor: Optional[QueryProcessor] = None,
        document_store: Optional[DocumentStore] = None,
        inverted_index: Optional[InvertedIndex] = None,
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
        self.total_docs = self.document_store.total_docs

        # Build TF-IDF matrix using sklearn
        self._build_tfidf_index()

    def _load_inverted_index(self) -> InvertedIndex:
        for path in ['data/index/inverted_index.pkl', 'data/index/index.pkl']:
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    print(f"⚠️ [VSMSearchService] {e}")
        index = InvertedIndex()
        for doc_id, doc in self.document_store.get_all_documents().items():
            index.add_document(doc_id, doc['tokens'])
        return index

    def _build_tfidf_index(self) -> None:
        """Fit TfidfVectorizer on pre-tokenized document corpus."""
        all_docs = self.document_store.get_all_documents()
        self.doc_ids = list(all_docs.keys())

        # Documents are already tokenized — join tokens back to string
        # so sklearn's analyzer splits them the same way
        doc_strings = [
            ' '.join(all_docs[doc_id]['tokens'])
            for doc_id in self.doc_ids
        ]

        # ✅ TfidfVectorizer من مكتبة sklearn
        # token_pattern=r'\S+' يتعامل مع أي token مهما كان شكله
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            token_pattern=r'\S+',
            smooth_idf=True,   # log((1+N)/(1+df)) + 1  — نفس Smoothed IDF
            sublinear_tf=False,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(doc_strings)
        # tfidf_matrix shape: (n_docs, n_terms) — sparse CSR matrix
        print(f"✅ [VSMSearchService] TF-IDF matrix built — "
              f"{self.tfidf_matrix.shape[0]} docs × {self.tfidf_matrix.shape[1]} terms")

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []

        # 1. Preprocess query (same pipeline as documents)
        query_tokens = self.query_processor.process_query(query)
        if not query_tokens:
            return []

        # 2. Transform query using the SAME fitted vectorizer
        query_str = ' '.join(query_tokens)
        query_vec = self.vectorizer.transform([query_str])  # sparse (1, n_terms)

        # 3. ✅ Cosine similarity من sklearn
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]  # (n_docs,)

        # 4. Top-k
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            doc_id = self.doc_ids[idx]
            doc = self.document_store.get_doc(doc_id)
            text = doc['text'] if doc else ""
            results.append({
                'doc_id': doc_id,
                'score': round(score, 6),
                'text': text[:300] + ('...' if len(text) > 300 else ''),
                'full_text': text,
                'method': 'VSM TF-IDF — sklearn TfidfVectorizer (Cosine Similarity)',
            })
        return results

    def get_term_details(self, doc_id: str, term: str) -> Dict[str, float]:
        """TF, IDF, TF-IDF لمصطلح في وثيقة — للـ UI."""
        try:
            term_idx = self.vectorizer.vocabulary_.get(term)
            doc_idx = self.doc_ids.index(doc_id)
            if term_idx is None:
                return {'tf': 0.0, 'idf': 0.0, 'tfidf': 0.0}
            tfidf_val = float(self.tfidf_matrix[doc_idx, term_idx])
            idf_val = float(self.vectorizer.idf_[term_idx])
            tf_val = tfidf_val / idf_val if idf_val else 0.0
            return {'tf': round(tf_val, 6), 'idf': round(idf_val, 6), 'tfidf': round(tfidf_val, 6)}
        except Exception:
            return {'tf': 0.0, 'idf': 0.0, 'tfidf': 0.0}