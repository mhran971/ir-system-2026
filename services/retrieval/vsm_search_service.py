# services/retrieval/vsm_search_service.py
import os
import pickle
import math
from typing import List, Dict, Any, Optional
from services.query_processing.query_processor import QueryProcessor
from services.indexing.document_store import DocumentStore
from services.indexing.inverted_index import InvertedIndex
from services.ranking.vsm_scorer import VSMScorer

class VSMSearchService:
    """
    Service layer orchestrator for the Vector Space Model (VSM).
    Coordinates QueryProcessor, InvertedIndex, and DocumentStore,
    and calculates Cosine Similarity using InvertedIndex postings.
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
        
        self.total_docs = self.inverted_index.total_documents
        
        # Precompute or load document vector norms
        self.doc_norms = self._load_or_compute_norms()

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
        
        # Fallback to build new index from document store
        print("⚠️ [VSMSearchService] No inverted index found on disk. Building from store...")
        index = InvertedIndex()
        for doc_id in self.document_store._doc_lengths.keys():
            doc = self.document_store.get_doc(doc_id)
            if doc:
                index.add_document(doc_id, doc.get('tokens', []))
        return index

    def _get_norms_path(self) -> str:
        default_path = 'data/index/vsm_doc_norms.pkl'
        try:
            import shutil
            abs_path = os.path.abspath(default_path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            drive = os.path.splitdrive(abs_path)[0]
            usage = shutil.disk_usage(drive)
            if usage.free < 0.1 * 1024 * 1024 * 1024: # Less than 100MB
                fallback_dir = os.path.expanduser('~/.ir_system_cache')
                os.makedirs(fallback_dir, exist_ok=True)
                path = os.path.join(fallback_dir, 'vsm_doc_norms.pkl')
                print(f"ℹ️ [VSMSearchService] Target drive space is low. Using C: drive fallback: {path}")
                return path
        except Exception as e:
            print(f"⚠️ [VSMSearchService] Disk space check failed: {e}")
        return default_path

    def _load_or_compute_norms(self) -> Dict[str, float]:
        """Loads VSM document vector norms from disk, or computes and caches them."""
        norms_path = self._get_norms_path()
        if os.path.exists(norms_path):
            try:
                with open(norms_path, 'rb') as f:
                    norms = pickle.load(f)
                print(f"✅ [VSMSearchService] Loaded precomputed VSM document norms from {norms_path}")
                return norms
            except Exception as e:
                print(f"⚠️ [VSMSearchService] Error loading norms from {norms_path}: {e}. Recomputing...")

        print("⏳ [VSMSearchService] Precomputing VSM document vector norms (runs once)...")
        from collections import defaultdict
        doc_norms = defaultdict(float)
        
        # 1. Compute IDF for all terms in vocabulary
        print("   Step 1/3: Calculating vocabulary IDFs...")
        idf_dict = VSMScorer.compute_idf(dict(self.inverted_index._doc_frequency), self.total_docs)
        
        # 2. Accumulate squared term weights for each document
        print("   Step 2/3: Accumulating squared weights for all documents...")
        for term, postings in self.inverted_index._index.items():
            idf = idf_dict.get(term, 0.0)
            if idf <= 0:
                continue
            for doc_id, count in postings.items():
                doc_len = self.inverted_index.get_document_length(doc_id)
                if doc_len > 0:
                    tf = count / doc_len
                    doc_norms[doc_id] += (tf * idf) ** 2
        
        # 3. Take square root to get Euclidean norm (L2 norm)
        print("   Step 3/3: Finalizing Euclidean norms...")
        for doc_id in list(doc_norms.keys()):
            doc_norms[doc_id] = math.sqrt(doc_norms[doc_id])
            
        # Save norms to disk
        os.makedirs(os.path.dirname(norms_path), exist_ok=True)
        with open(norms_path, 'wb') as f:
            pickle.dump(dict(doc_norms), f)
        print(f"✅ [VSMSearchService] Precomputed norms saved to {norms_path}")
        return dict(doc_norms)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Executes query retrieval and VSM Cosine Similarity ranking using sparse InvertedIndex.
        """
        if not query or not query.strip():
            return []

        # 1. Query processing
        query_tokens = self.query_processor.process_query(query)
        if not query_tokens:
            return []
        
        # 2. Count term frequencies in query
        from collections import Counter
        query_counts = Counter(query_tokens)
        
        # 3. Calculate query term weights and query norm
        query_weights = {}
        query_sum_sq = 0.0
        
        # Calculate IDF values for query tokens
        q_idf_dict = VSMScorer.compute_idf(
            {term: self.inverted_index._doc_frequency.get(term, 0) for term in query_counts.keys()},
            self.total_docs
        )
        
        for term, q_tf in query_counts.items():
            idf = q_idf_dict.get(term, 0.0)
            weight = q_tf * idf
            query_weights[term] = weight
            query_sum_sq += weight ** 2
            
        query_norm = math.sqrt(query_sum_sq)
        if query_norm == 0:
            return []
            
        # 4. Calculate dot products for candidate documents
        dot_products = {}
        for term, q_weight in query_weights.items():
            if q_weight == 0:
                continue
            postings = self.inverted_index.get_documents_for_term(term)
            idf = q_idf_dict.get(term, 0.0)
            
            for doc_id, count in postings.items():
                doc_len = self.inverted_index.get_document_length(doc_id)
                if doc_len > 0:
                    doc_tf = count / doc_len
                    doc_weight = doc_tf * idf
                    dot_products[doc_id] = dot_products.get(doc_id, 0.0) + (doc_weight * q_weight)
                    
        # 5. Compute cosine similarities
        scores = []
        for doc_id, dot_prod in dot_products.items():
            doc_norm = self.doc_norms.get(doc_id, 0.0)
            if doc_norm > 0:
                sim = dot_prod / (doc_norm * query_norm)
                if sim > 0:
                    scores.append((doc_id, sim))
                    
        # 6. Sort and get top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        top_scores = scores[:top_k]
        
        # 7. Formulate results
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
        """Get detailed TF, IDF, and TF-IDF values for a term in a document using InvertedIndex."""
        doc_len = self.inverted_index.get_document_length(doc_id)
        if doc_len == 0:
            return {'tf': 0.0, 'idf': 0.0, 'tfidf': 0.0}
            
        term_count = self.inverted_index.get_term_frequency(term, doc_id)
        tf = term_count / doc_len
        
        # Calculate IDF
        df = self.inverted_index._doc_frequency.get(term, 0)
        idf_dict = VSMScorer.compute_idf({term: df}, self.total_docs)
        idf = idf_dict.get(term, 0.0)
        
        tfidf = tf * idf
        return {'tf': tf, 'idf': idf, 'tfidf': tfidf}