"""Pydantic request and response schemas for the IR System API."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ── Search ─────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    model: Literal["bm25", "vsm", "bert", "hybrid_serial", "hybrid_parallel", "simple"] = "bm25"
    top_k: int = Field(10, ge=1, le=200)

    # BM25 / Hybrid BM25 params
    k1: float = Field(1.5, gt=0.0, le=10.0)
    b: float = Field(0.75, ge=0.0, le=1.0)

    # Hybrid serial
    bm25_candidates: int = Field(100, ge=10, le=500)

    # Hybrid parallel
    fusion: Literal["rrf", "linear"] = "rrf"
    bm25_weight: float = Field(0.4, ge=0.0, le=1.0)
    bert_weight: float = Field(0.6, ge=0.0, le=1.0)

    # Query refinement
    use_spelling: bool = True
    use_synonyms: bool = True
    use_prf: bool = True
    use_history: bool = True
    num_prf_terms: int = Field(3, ge=1, le=10)

    # Clustering
    cluster_results: bool = False
    n_clusters: int = Field(4, ge=2, le=10)


class SearchResult(BaseModel):
    doc_id: str
    score: float
    text: str
    full_text: str
    method: Optional[str] = None
    bm25_rank: Optional[Any] = None
    bert_rank: Optional[Any] = None


class ClusterData(BaseModel):
    scatter_data: List[Dict[str, Any]]
    cluster_labels: Dict[str, Any]
    grouped_results: Dict[str, Any]


class SearchResponse(BaseModel):
    results: List[SearchResult]
    refined_query: str
    refinement_info: Dict[str, Any]
    method_used: str
    elapsed: float
    cluster_data: Optional[ClusterData] = None


# ── Refine ─────────────────────────────────────────────────────────────────────

class RefineRequest(BaseModel):
    query: str
    use_spelling: bool = True
    use_synonyms: bool = True
    use_prf: bool = True
    use_history: bool = False
    num_prf_terms: int = Field(3, ge=1, le=10)
    # Caller may supply pre-fetched PRF docs; if absent and use_prf=True the
    # endpoint fetches them via BM25 internally.
    prf_docs: Optional[List[Dict[str, Any]]] = None


class RefineResponse(BaseModel):
    original: str
    corrected: str
    tokens: List[str]
    weights: Dict[str, float]
    expanded_query: str
    prf_terms_added: List[str]
    synonyms_added: List[str]
    history_boost_applied: Dict[str, float]


# ── Term details ────────────────────────────────────────────────────────────────

class TermDetailsRequest(BaseModel):
    doc_id: str
    term: str
    model: Literal["bm25", "vsm"]


class TermDetailsResponse(BaseModel):
    doc_id: str
    term: str
    model: str
    details: Dict[str, Any]


# ── Document ────────────────────────────────────────────────────────────────────

class DocumentResponse(BaseModel):
    doc_id: str
    text: str


# ── Evaluation ─────────────────────────────────────────────────────────────────

class EvaluateRunResponse(BaseModel):
    status: Literal["completed", "error"]
    stdout: str
    stderr: str
    returncode: int
