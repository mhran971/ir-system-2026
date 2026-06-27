"""
Singleton service container for the IR System API.
Mirrors the @st.cache_resource loaders in ui/app.py.
All services are initialized once during FastAPI lifespan startup.
"""
import threading

# ── Singletons ────────────────────────────────────────────────────────────────
_simple_svc = None
_vsm_svc = None
_bm25_svc = None
_bert_svc = None
_hybrid_svc = None
_refiner = None
_clustering_svc = None

# Guards for mutable shared state on the singleton services.
# BM25/Hybrid: k1 and b are mutated per-request before search.
# Refiner: user_history_terms is mutated by update_history().
_bm25_lock = threading.Lock()
_history_lock = threading.Lock()


def init_all():
    """Load every service into memory. Called once at API startup."""
    global _simple_svc, _vsm_svc, _bm25_svc, _bert_svc
    global _hybrid_svc, _refiner, _clustering_svc

    from services.retrieval.search_service import SearchService
    from services.retrieval.vsm_search_service import VSMSearchService
    from services.retrieval.bm25_search_service import BM25SearchService
    from services.ranking.embeddings.bert_search_service import BERTSearchService
    from services.ranking.hybrid.hybrid_search_service import HybridSearchService
    from services.query_processing.query_refiner import QueryRefiner
    from services.clustering.clustering_service import ClusteringService

    # BM25 must be first — refiner and hybrid depend on it.
    print("Loading BM25...")
    _bm25_svc = BM25SearchService(k1=1.5, b=0.75)

    print("Loading VSM...")
    _vsm_svc = VSMSearchService()

    print("Loading Simple TF-IDF...")
    _simple_svc = SearchService()

    print("Loading BERT...")
    _bert_svc = BERTSearchService(model_key="fast")

    print("Loading Hybrid...")
    _hybrid_svc = HybridSearchService(
        bm25_service=_bm25_svc,
        bert_service=_bert_svc,
    )

    print("Loading QueryRefiner...")
    try:
        known_terms = set(_bm25_svc.inverted_index.doc_frequency.keys())
        term_freqs = dict(_bm25_svc.inverted_index.doc_frequency)
        total_docs = _bm25_svc.document_store.total_docs
        _refiner = QueryRefiner(
            known_terms=known_terms,
            term_frequencies=term_freqs,
            total_docs=total_docs,
        )
        print(f"  Loaded {len(known_terms)} known terms for QueryRefiner")
    except Exception as exc:
        print(f"  Warning: refiner fallback — {exc}")
        _refiner = QueryRefiner()

    print("Loading ClusteringService...")
    _clustering_svc = ClusteringService(bert_service=_bert_svc)

    print("All services loaded.")


# ── Accessors ─────────────────────────────────────────────────────────────────

def get_simple():
    return _simple_svc

def get_vsm():
    return _vsm_svc

def get_bm25():
    return _bm25_svc

def get_bert():
    return _bert_svc

def get_hybrid():
    return _hybrid_svc

def get_refiner():
    return _refiner

def get_clustering():
    return _clustering_svc
