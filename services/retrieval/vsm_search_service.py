# services/retrieval/vsm_search_service.py
import os
import pickle
from typing import List, Dict, Any, Optional
from services.query_processing.query_processor import QueryProcessor
from services.indexing.document_store import DocumentStore
from services.indexing.inverted_index import InvertedIndex
from sklearn.feature_extraction.text import TfidfVectorizer

class VSMSearchService:
    """
    Service layer orchestrator for the Vector Space Model (VSM).
    Coordinates QueryProcessor, InvertedIndex, and DocumentStore,
    and applies TfidfVectorizer from sklearn.
    """
    def __init__(
        self,
        query_processor: Optional[QueryProcessor] = None,
        document_store: Optional[DocumentStore] = None,
        inverted_index: Optional[InvertedIndex] = None
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
        
        self.doc_ids = list(self.document_store.get_all_documents().keys())
        self.total_docs = len(self.doc_ids)
        
        if self.total_docs > 0:
            tokenized_corpus = [self.document_store.get_doc(doc_id)['tokens'] for doc_id in self.doc_ids]
            
            def identity_tokenizer(text):
                return text
                
            self.vectorizer = TfidfVectorizer(
                tokenizer=identity_tokenizer,
                preprocessor=identity_tokenizer,
                token_pattern=None
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(tokenized_corpus)
        else:
            self.vectorizer = None
            self.tfidf_matrix = None

        # Build idf_values for compatibility
        self.idf_values: Dict[str, float] = {}
        if self.vectorizer and hasattr(self.vectorizer, 'vocabulary_'):
            for term, idx in self.vectorizer.vocabulary_.items():
                self.idf_values[term] = float(self.vectorizer.idf_[idx])

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
                    print(f"✅ [VSMSearchService] Loaded InvertedIndex from {path}")
                    return index
                except Exception as e:
                    print(f"⚠️ [VSMSearchService] Error loading index from {path}: {e}")
        
        # Fallback to build new index from document store if not found
        print("⚠️ [VSMSearchService] No inverted index found on disk. Building from store...")
        index = InvertedIndex()
        for doc_id, doc in self.document_store.get_all_documents().items():
            index.add_document(doc_id, doc['tokens'])
        return index

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Executes query retrieval and VSM ranking.
        """
        # Handle empty/whitespace-only queries
        if not query or not query.strip():
            return []

        # 1. Query processing
        query_tokens = self.query_processor.process_query(query)
        if not query_tokens or self.tfidf_matrix is None:
            return []
        
        # 2. Vectorize the query
        query_vector = self.vectorizer.transform([query_tokens])
        
        # 3. Calculate cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(self.tfidf_matrix, query_vector).flatten()
        
        # 4. Associate scores with doc_ids
        scores = []
        for idx, score in enumerate(similarities):
            if score > 0:
                scores.append((self.doc_ids[idx], float(score)))
                    
        # 5. Sort and return top-k
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
                'method': 'VSM TF-IDF (Cosine Similarity)'
            })
        return results

    def get_term_details(self, doc_id: str, term: str) -> Dict[str, float]:
        """Get detailed TF, IDF, and TF-IDF values for a term in a document."""
        doc = self.document_store.get_doc(doc_id)
        if not doc or self.vectorizer is None:
            return {'tf': 0.0, 'idf': 0.0, 'tfidf': 0.0}
        
        tokens = doc.get('tokens', [])
        
        # Calculate IDF from vectorizer
        term_idx = self.vectorizer.vocabulary_.get(term)
        idf = float(self.vectorizer.idf_[term_idx]) if term_idx is not None else 0.0
        
        # Calculate TF
        tf = 0.0
        term_count = tokens.count(term)
        if term_count > 0 and len(tokens) > 0:
            tf = term_count / len(tokens)
            
        tfidf = tf * idf
            
        return {'tf': tf, 'idf': idf, 'tfidf': tfidf}