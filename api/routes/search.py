import time
from fastapi import APIRouter, HTTPException
from api import services
from api.models import ClusterData, SearchRequest, SearchResponse, SearchResult

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """
    Execute a search query using the specified retrieval model.
    Mirrors the search flow in ui/app.py lines 199-291.
    """
    refiner = services.get_refiner()
    if refiner is None:
        raise HTTPException(status_code=503, detail="Services not initialized")

    t0 = time.time()

    # ── Step 1: history update ────────────────────────────────────────────────
    if req.use_history:
        with services._history_lock:
            refiner.update_history(req.query)

    # ── Step 2: PRF pre-fetch ─────────────────────────────────────────────────
    prf_docs = []
    if req.use_prf:
        bm25 = services.get_bm25()
        if bm25 is None:
            raise HTTPException(status_code=503, detail="BM25 service not initialized")
        prf_docs = bm25.search(req.query, top_k=5)

    # ── Step 3: query refinement ──────────────────────────────────────────────
    refined = refiner.refine(
        query=req.query,
        apply_spelling=req.use_spelling,
        apply_synonyms=req.use_synonyms,
        apply_prf=req.use_prf,
        apply_history=req.use_history,
        top_docs=prf_docs,
        num_prf_terms=req.num_prf_terms,
        synonym_limit=1,
    )
    expanded_query = refined["expanded_query"]
    refinement_info = refined

    # ── Step 4: model dispatch ────────────────────────────────────────────────
    results = []
    method_used = req.model

    if req.model == "bm25":
        svc = services.get_bm25()
        if svc is None:
            raise HTTPException(status_code=503, detail="BM25 service not initialized")
        with services._bm25_lock:
            svc.k1 = req.k1
            svc.b = req.b
            results = svc.search(expanded_query, top_k=req.top_k)
        method_used = f"BM25 (k1={req.k1}, b={req.b}) + Refinement"

    elif req.model == "hybrid_serial":
        svc = services.get_hybrid()
        if svc is None:
            raise HTTPException(status_code=503, detail="Hybrid service not initialized")
        with services._bm25_lock:
            svc.k1 = req.k1
            svc.b = req.b
            results = svc.search(
                expanded_query,
                mode="serial",
                top_k=req.top_k,
                bm25_candidates=req.bm25_candidates,
            )
        method_used = "Hybrid Serial + Refinement"

    elif req.model == "hybrid_parallel":
        svc = services.get_hybrid()
        if svc is None:
            raise HTTPException(status_code=503, detail="Hybrid service not initialized")
        with services._bm25_lock:
            svc.k1 = req.k1
            svc.b = req.b
            results = svc.search(
                expanded_query,
                mode="parallel",
                fusion=req.fusion,
                top_k=req.top_k,
                bm25_weight=req.bm25_weight,
                bert_weight=req.bert_weight,
            )
        method_used = f"Hybrid Parallel {req.fusion.upper()} + Refinement"

    elif req.model == "vsm":
        svc = services.get_vsm()
        if svc is None:
            raise HTTPException(status_code=503, detail="VSM service not initialized")
        results = svc.search(expanded_query, top_k=req.top_k)
        method_used = "VSM TF-IDF + Refinement"

    elif req.model == "bert":
        svc = services.get_bert()
        if svc is None:
            raise HTTPException(status_code=503, detail="BERT service not initialized")
        results = svc.search(expanded_query, top_k=req.top_k)
        method_used = "BERT Semantic + Refinement"

    else:  # simple
        svc = services.get_simple()
        if svc is None:
            raise HTTPException(status_code=503, detail="Simple service not initialized")
        results = svc.search(expanded_query, top_k=req.top_k)
        method_used = "Simple TF-IDF + Refinement"

    # ── Step 5: attach refinement info (mirrors app.py line 274-275) ─────────
    for r in results:
        r["refinement_info"] = refinement_info

    elapsed = time.time() - t0

    # ── Step 6: optional clustering ───────────────────────────────────────────
    cluster_data = None
    if req.cluster_results and results:
        clustering_svc = services.get_clustering()
        if clustering_svc is not None:
            raw = clustering_svc.cluster_search_results(results, n_clusters=req.n_clusters)
            cluster_data = ClusterData(
                scatter_data=raw.get("scatter_data", []),
                cluster_labels={str(k): v for k, v in raw.get("cluster_labels", {}).items()},
                grouped_results={str(k): v for k, v in raw.get("grouped_results", {}).items()},
            )

    # ── Build response ────────────────────────────────────────────────────────
    search_results = [
        SearchResult(
            doc_id=r.get("doc_id", ""),
            score=float(r.get("score", 0.0)),
            text=r.get("text", ""),
            full_text=r.get("full_text", ""),
            method=r.get("method"),
            bm25_rank=r.get("bm25_rank"),
            bert_rank=r.get("bert_rank"),
        )
        for r in results
    ]

    return SearchResponse(
        results=search_results,
        refined_query=expanded_query,
        refinement_info={
            k: v for k, v in refinement_info.items()
            if k != "refinement_info"  # avoid nesting the full dict
        },
        method_used=method_used,
        elapsed=elapsed,
        cluster_data=cluster_data,
    )
