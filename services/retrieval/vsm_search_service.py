# services/retrieval/vsm_search_service.py
import os
import pickle
from typing import List, Dict, Any, Optional
from services.query_processing.query_processor import QueryProcessor
from services.indexing.document_store import DocumentStore
from services.indexing.inverted_index import InvertedIndex
from services.ranking.vsm_scorer import VSMScorer

class VSMSearchService:
    """
    Service layer orchestrator for the Vector Space Model (VSM).
    Coordinates QueryProcessor, InvertedIndex, and DocumentStore,
    and applies VSMScorer mathematical ranking.
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
        
        # Document representations
        self.doc_vectors: Dict[str, Dict[str, float]] = {}
        self.idf_values: Dict[str, float] = {}
        self._build_document_vectors()
        self.total_docs = self.document_store.total_docs

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

    def _build_document_vectors(self) -> None:
        total_docs = self.document_store.total_docs
        if total_docs == 0:
            return
        
        # Compute IDF using VSMScorer
        self.idf_values = VSMScorer.compute_idf(
            doc_frequency=self.inverted_index.doc_frequency,
            total_docs=total_docs,
            smoothing=True
        )
        
        # Compute document TF-IDF vectors
        for doc_id, doc in self.document_store.get_all_documents().items():
            tf = VSMScorer.compute_tf(doc['tokens'], normalize=True)
            self.doc_vectors[doc_id] = VSMScorer.compute_tfidf(tf, self.idf_values)
        print(f"✅ [VSMSearchService] Precomputed TF-IDF vectors for {len(self.doc_vectors)} documents.")

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Executes query retrieval and VSM ranking.
        """
        # 1. Query processing
        query_tokens = self.query_processor.process_query(query)
        if not query_tokens:
            return []
        
        # 2. Candidate retrieval
        candidates = self.inverted_index.get_candidates(query_tokens)
        
        # 3. Query vector generation
        query_tf = VSMScorer.compute_tf(query_tokens, normalize=True)
        query_vector = VSMScorer.compute_tfidf(query_tf, self.idf_values)
        
        # 4. Cosine similarity scoring
        scores = []
        for doc_id in candidates:
            doc_vector = self.doc_vectors.get(doc_id)
            if doc_vector:
                score = VSMScorer.cosine_similarity(query_vector, doc_vector)
                if score > 0:
                    scores.append((doc_id, score))
                    
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
        if not doc:
            return {'tf': 0.0, 'idf': 0.0, 'tfidf': 0.0}
        
        tokens = doc.get('tokens', [])
        idf = self.idf_values.get(term, 0.0)
        
        # Calculate TF
        tf = 0.0
        tfidf = 0.0
        term_count = tokens.count(term)
        if term_count > 0 and len(tokens) > 0:
            tf = term_count / len(tokens)
            tfidf = tf * idf
            
        return {'tf': tf, 'idf': idf, 'tfidf': tfidf}