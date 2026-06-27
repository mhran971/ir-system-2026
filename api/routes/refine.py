from fastapi import APIRouter, HTTPException
from api import services
from api.models import RefineRequest, RefineResponse

router = APIRouter(tags=["query"])


@router.post("/refine", response_model=RefineResponse)
def refine_query(req: RefineRequest):
    """
    Run the 4-stage query refinement pipeline without executing a search.
    If use_prf is True and no prf_docs are supplied, BM25 is used internally
    to fetch the top-5 candidate documents for pseudo-relevance feedback.
    """
    refiner = services.get_refiner()
    if refiner is None:
        raise HTTPException(status_code=503, detail="Refiner service not initialized")

    prf_docs = req.prf_docs or []

    if req.use_prf and not prf_docs:
        bm25 = services.get_bm25()
        if bm25 is None:
            raise HTTPException(status_code=503, detail="BM25 service not initialized")
        prf_docs = bm25.search(req.query, top_k=5)

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

    return RefineResponse(
        original=refined.get("original", req.query),
        corrected=refined.get("corrected", req.query),
        tokens=refined.get("tokens", []),
        weights=refined.get("weights", {}),
        expanded_query=refined.get("expanded_query", req.query),
        prf_terms_added=refined.get("prf_terms_added", []),
        synonyms_added=refined.get("synonyms_added", []),
        history_boost_applied=refined.get("history_boost_applied", {}),
    )
