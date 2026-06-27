from fastapi import APIRouter, HTTPException
from typing import Any, Dict
from api import services

router = APIRouter(tags=["models"])

_VALID_MODELS = {"bm25", "vsm", "bert", "hybrid", "simple"}


@router.get("/models/{model}/stats")
def get_model_stats(model: str) -> Dict[str, Any]:
    """Return index/model statistics for the requested retrieval model."""
    if model not in _VALID_MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model '{model}'. Valid options: {sorted(_VALID_MODELS)}",
        )

    if model == "bm25":
        svc = services.get_bm25()
        if svc is None:
            raise HTTPException(status_code=503, detail="BM25 service not initialized")
        stats = svc.get_stats()
        stats["k1"] = svc.k1
        stats["b"] = svc.b
        return stats

    if model == "vsm":
        svc = services.get_vsm()
        if svc is None:
            raise HTTPException(status_code=503, detail="VSM service not initialized")
        return {
            "total_documents": svc.total_docs,
            "unique_terms": len(svc.inverted_index),
        }

    if model == "bert":
        svc = services.get_bert()
        if svc is None:
            raise HTTPException(status_code=503, detail="BERT service not initialized")
        return svc.get_stats()

    if model == "hybrid":
        svc = services.get_hybrid()
        if svc is None:
            raise HTTPException(status_code=503, detail="Hybrid service not initialized")
        return svc.get_stats()

    # simple
    svc = services.get_simple()
    if svc is None:
        raise HTTPException(status_code=503, detail="Simple service not initialized")
    return {
        "total_documents": len(svc.documents),
        "unique_terms": len(svc.index) if svc.index else 0,
    }
