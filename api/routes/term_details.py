from fastapi import APIRouter, HTTPException
from api import services
from api.models import TermDetailsRequest, TermDetailsResponse

router = APIRouter(tags=["search"])


@router.post("/term-details", response_model=TermDetailsResponse)
def term_details(req: TermDetailsRequest):
    """
    Return per-term scoring details for a document.
    - bm25: returns {tf, idf, score}  via BM25SearchService.get_term_score()
    - vsm:  returns {tf, idf, tfidf}  via VSMSearchService.get_term_details()
    """
    if req.model == "bm25":
        svc = services.get_bm25()
        if svc is None:
            raise HTTPException(status_code=503, detail="BM25 service not initialized")
        details = svc.get_term_score(req.doc_id, req.term)

    elif req.model == "vsm":
        svc = services.get_vsm()
        if svc is None:
            raise HTTPException(status_code=503, detail="VSM service not initialized")
        details = svc.get_term_details(req.doc_id, req.term)

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{req.model}' does not support per-term scoring. Use 'bm25' or 'vsm'.",
        )

    return TermDetailsResponse(
        doc_id=req.doc_id,
        term=req.term,
        model=req.model,
        details=details or {},
    )
