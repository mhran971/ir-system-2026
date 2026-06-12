# services/ranking/vsm_tfidf_custom.py
"""
Vector Space Model with TF-IDF - Main Search Engine.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, List, Tuple, Set, Any, Optional

from services.preprocessing.preprocessor import TextPreprocessor
from services.indexing.inverted_index import InvertedIndex

from .config import VSMTFIDFConfig
from .models import SearchResult, TermDetails
from .tfidf_calculator import TFIDFCalculator
from .cosine_similarity import CosineSimilarity
from .document_store import DocumentStore


class VSM_TFIDF_Custom:
    """
    Vector Space Model with TF-IDF and Cosine Similarity.
    
    A clean, modular implementation of classic information retrieval.
    """
    
    def __init__(self, config: Optional[VSMTFIDFConfig] = None) -> None:
        """
        Initialize the VSM TF-IDF search engine.
        
        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or VSMTFIDFConfig()
        
        # Initialize components
        self.preprocessor = TextPreprocessor(
            use_stemming=self.config.USE_STEMMING,
            use_lemmatization=self.config.USE_LEMMATIZATION
        )
        self.doc_store = DocumentStore(self.preprocessor)
        self.inverted_index = InvertedIndex()
        self.tfidf_calc = TFIDFCalculator()
        self.similarity = CosineSimilarity()
        
        # Vectors storage
        self.doc_vectors: Dict[str, Dict[str, float]] = {}
        self.idf_values: Dict[str, float] = {}
        
        # ✅ إضافة total_docs كـ attribute عادي
        self.total_docs = 0
        
        # Load and build
        self._initialize()
    
    @property
    def total_docs_property(self) -> int:
        """Get total number of documents (as property)."""
        return self.doc_store.total_docs
    
    @property
    def doc_count(self) -> int:
        """Alternative name for total documents."""
        return self.doc_store.total_docs
    
    def _initialize(self) -> None:
        """Load data and build all indexes."""
        # Load documents
        num_docs = self.doc_store.load_from_pickle(self.config.DATA_PATHS)
        
        # ✅ تحديث total_docs
        self.total_docs = self.doc_store.total_docs
        
        print(f"✅ Loaded {num_docs} documents")
        print(f"   Total docs: {self.total_docs}")  # للتحقق
        
        if num_docs == 0:
            print("⚠️ Warning: No documents loaded!")
            return
        
        # Build inverted index
        self._build_inverted_index()
        
        # Compute IDF
        self._compute_idf()
        
        # Build document vectors
        self._build_document_vectors()
    
    def _build_inverted_index(self) -> None:
        """Build inverted index from loaded documents."""
        print("\n🔨 Building inverted index...")
        
        for doc_id in self.doc_store.get_all_ids():
            tokens = self.doc_store.get_tokens(doc_id)
            self.inverted_index.add_document(doc_id, tokens)
        
        stats = self.inverted_index.get_stats()
        print(f"✅ Built index: {stats['unique_terms']} unique terms, {stats['total_documents']} docs")
    
    def _compute_idf(self) -> None:
        """Compute IDF values for all terms."""
        print("\n📊 Computing IDF values...")
        
        # ✅ استخدم self.total_docs (الـ attribute) أو self.doc_store.total_docs
        self.idf_values = self.tfidf_calc.compute_idf(
            doc_frequency=self.inverted_index.doc_frequency,
            total_docs=self.total_docs,  # ✅ الآن self.total_docs موجود
            smoothing=self.config.IDF_SMOOTHING
        )
        
        self._print_idf_stats()
    
    def _print_idf_stats(self) -> None:
        """Print statistics about IDF values."""
        if not self.idf_values:
            return
        
        sorted_terms = sorted(
            self.idf_values.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        print(f"   ✅ IDF computed for {len(self.idf_values)} terms")
        if sorted_terms:
            print(f"   📈 Highest IDF (rare): {sorted_terms[:3]}")
            print(f"   📉 Lowest IDF (common): {sorted_terms[-3:]}")
    
    def _build_document_vectors(self) -> None:
        """Build TF-IDF vectors for all documents."""
        print("\n🔨 Building document vectors...")
        
        for doc_id, doc in self.doc_store.get_all_documents().items():
            tf = self.tfidf_calc.compute_tf(
                doc.tokens,
                normalize=self.config.TF_LENGTH_NORMALIZATION
            )
            tfidf = self.tfidf_calc.compute_tfidf(tf, self.idf_values)
            self.doc_vectors[doc_id] = tfidf
        
        print(f"✅ Built vectors for {len(self.doc_vectors)} documents")
    
    def _compute_query_vector(self, query: str) -> Tuple[Dict[str, float], List[str]]:
        """Compute TF-IDF vector for a query."""
        query_tokens = self.preprocessor.process(query)
        
        if not query_tokens:
            return {}, []
        
        query_tf = self.tfidf_calc.compute_tf(
            query_tokens,
            normalize=self.config.TF_LENGTH_NORMALIZATION
        )
        query_vector = self.tfidf_calc.compute_tfidf(query_tf, self.idf_values)
        
        return query_vector, query_tokens
    
    def search(
        self, 
        query: str, 
        top_k: Optional[int] = None, 
        use_tfidf: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for documents matching the query.
        
        Args:
            query: Search query string
            top_k: Number of results to return (uses config default if None)
            use_tfidf: Use TF-IDF (True) or just TF (False)
        
        Returns:
            List of search result dictionaries
        """
        top_k = top_k or self.config.DEFAULT_TOP_K
        
        print(f"\n🔍 Query: '{query}'")
        
        # Compute query vector
        query_vector, query_tokens = self._compute_query_vector(query)
        
        print(f"   Tokens: {query_tokens}")
        print(f"   Unique terms: {len(query_vector)}")
        
        if not query_vector:
            return []
        
        # Get candidate documents
        candidates = self.inverted_index.get_candidates(query_tokens)
        
        # ✅ استخدام self.total_docs (الموجود الآن)
        print(f"   Candidates: {len(candidates)} (from {self.total_docs} total)")
        
        # Score candidates
        results = self._score_candidates(query_vector, candidates, use_tfidf)
        
        # Sort and return top results
        results.sort(key=lambda x: x.score, reverse=True)
        top_results = results[:top_k]
        
        print(f"✅ Found {len(top_results)} results")
        return [r.to_dict() for r in top_results]
    
    def _score_candidates(
        self,
        query_vector: Dict[str, float],
        candidates: Set[str],
        use_tfidf: bool
    ) -> List[SearchResult]:
        """Score candidate documents."""
        results = []
        
        for doc_id in candidates:
            score = self._calculate_score(doc_id, query_vector, use_tfidf)
            
            if score > 0:
                result = self._create_result(doc_id, score, use_tfidf)
                results.append(result)
        
        return results
    
    def _calculate_score(
        self,
        doc_id: str,
        query_vector: Dict[str, float],
        use_tfidf: bool
    ) -> float:
        """Calculate similarity score for a document."""
        if use_tfidf and doc_id in self.doc_vectors:
            return self.similarity.compute(query_vector, self.doc_vectors[doc_id])
        
        if not use_tfidf:
            doc_tokens = self.doc_store.get_tokens(doc_id)
            doc_tf = self.tfidf_calc.compute_tf(
                doc_tokens,
                normalize=self.config.TF_LENGTH_NORMALIZATION
            )
            return self.similarity.compute(query_vector, doc_tf)
        
        return 0.0
    
    def _create_result(self, doc_id: str, score: float, use_tfidf: bool) -> SearchResult:
        """Create a search result object."""
        doc_text = self.doc_store.get_text(doc_id)
        preview = doc_text[:self.config.TEXT_PREVIEW_LENGTH]
        
        if len(doc_text) > self.config.TEXT_PREVIEW_LENGTH:
            preview += "..."
        
        return SearchResult(
            doc_id=doc_id,
            score=score,
            text=preview,
            full_text=doc_text,
            method="VSM_TFIDF" if use_tfidf else "VSM_TF_ONLY"
        )
    
    def get_term_details(self, doc_id: str, term: str) -> Dict[str, float]:
        """Get detailed TF, IDF, and TF-IDF values for a term in a document."""
        doc = self.doc_store.get(doc_id)
        if not doc:
            return TermDetails().to_dict()
        
        idf = self.idf_values.get(term, 0.0)
        details = self.tfidf_calc.compute_term_details(term, doc.tokens, idf)
        
        return details
    
    def get_document_terms(self, doc_id: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get top N terms from document vector."""
        if doc_id not in self.doc_vectors:
            return []
        
        sorted_terms = sorted(
            self.doc_vectors[doc_id].items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_terms[:top_n]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the search engine."""
        return {
            'total_documents': self.total_docs,  # ✅ الآن موجود
            'unique_terms': len(self.idf_values),
            'vectors_built': len(self.doc_vectors),
            'inverted_index_stats': self.inverted_index.get_stats(),
            'config': self.config.to_dict()
        }
    
    @staticmethod
    def get_parameters_explanation() -> Dict[str, Dict[str, str]]:
        """Return explanation of model parameters."""
        return {
            'tf_method': {
                'name': 'Length-normalized TF',
                'formula': 'tf(t,d) = count(t,d) / |d|',
                'why': 'Normalizes for document length'
            },
            'idf_method': {
                'name': 'Smoothed IDF',
                'formula': 'idf(t) = log10((N+1)/(df(t)+1)) + 1',
                'why': 'Prevents division by zero and gives balanced weights'
            },
            'similarity': {
                'name': 'Cosine Similarity',
                'formula': 'cos(q,d) = (q·d) / (||q|| × ||d||)',
                'why': 'Measures angle between vectors, ignoring length'
            }
        }


def main():
    """Interactive test function."""
    try:
        vsm = VSM_TFIDF_Custom()
        
        print("\n" + "=" * 60)
        print("🔍 VSM_TFIDF Search Test (Fixed Implementation)")
        print("=" * 60)
        
        print("\n📚 Statistics:")
        stats = vsm.get_stats()
        print(f"   Documents: {stats['total_documents']}")
        print(f"   Unique terms: {stats['unique_terms']}")
        print(f"   Vectors built: {stats['vectors_built']}")
        
        print("\n📚 Parameter Explanation:")
        params = vsm.get_parameters_explanation()
        for key, info in params.items():
            print(f"\n   {key.upper()}:")
            print(f"      Method: {info['name']}")
            print(f"      Formula: {info['formula']}")
            print(f"      Why: {info['why']}")
        
        while True:
            query = input("\n🔍 Enter search query (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
            
            if not query.strip():
                print("   Please enter a valid query.")
                continue
            
            results = vsm.search(query, top_k=5)
            
            if results:
                print("\n📊 Results:")
                for i, r in enumerate(results, 1):
                    print(f"\n{i}. Document {r['doc_id']} (Score: {r['score']:.4f})")
                    print(f"   Method: {r['method']}")
                    if r['text']:
                        print(f"   📄 {r['text'][:200]}...")
            else:
                print("   No results found.")
            print("-" * 40)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()