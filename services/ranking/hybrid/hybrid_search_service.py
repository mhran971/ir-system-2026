# services/ranking/hybrid/hybrid_search_service.py
"""
Hybrid Search Service — Serial and Parallel modes.

Serial:  BM25 retrieves candidates → BERT re-ranks via weighted combination
Parallel: BM25 + BERT run independently → Weighted RRF or linear fusion
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[3]))

from services.retrieval.bm25_search_service import BM25SearchService
from services.ranking.embeddings.bert_search_service import BERTSearchService

RRF_K = 60


class HybridSearchService:
    """
    Hybrid retrieval combining BM25 (sparse) and BERT (dense) models.

    Modes:
        serial   - BM25 first-stage then BERT re-ranks top candidates
        parallel - BM25 + BERT run simultaneously then scores fused

    Fusion methods (parallel only):
        rrf    - Weighted Reciprocal Rank Fusion (rank-based, robust)
        linear - Weighted linear combination of normalized scores
    """

    def __init__(
        self,
        bm25_service: Optional[BM25SearchService] = None,
        bert_service: Optional[BERTSearchService] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        print("Initializing HybridSearchService...")
        self.bm25 = bm25_service or BM25SearchService(k1=k1, b=b)
        self.bert = bert_service or BERTSearchService(model_key="fast")
        print(f"HybridSearchService ready — BM25: {self.bm25.document_store.total_docs:,} docs | BERT: {self.bert.total_docs:,} docs")

    # ── Public API ────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        mode: str = "serial",
        fusion: str = "rrf",
        top_k: int = 10,
        bm25_candidates: int = 100,
        bm25_weight: float = 0.6,
        bert_weight: float = 0.4,
        bert_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Args:
            query:      Query string (may be BM25-expanded for refinement)
            bert_query: Original un-expanded query for BERT encoding.
                        When provided, BERT uses this instead of query so its
                        embedding stays focused while BM25 benefits from expansion.
        """
        if not query or not query.strip():
            return []

        print(f"\nHybrid Search [{mode.upper()}] query='{query}'")

        if mode == "serial":
            return self._serial_search(query, top_k, bm25_candidates, bm25_weight, bert_weight, bert_query)
        elif mode == "parallel":
            return self._parallel_search(query, top_k, fusion, bm25_weight, bert_weight, bert_query)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'serial' or 'parallel'.")

    # ── Serial ────────────────────────────────────────────────────────────────

    def _serial_search(
        self,
        query: str,
        top_k: int,
        bm25_candidates: int,
        bm25_weight: float = 0.6,
        bert_weight: float = 0.4,
        bert_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Stage 1: BM25 retrieves a candidate pool larger than top_k.
        Stage 2: BERT scores all candidates.
        Final:   Weighted combination of min-max normalized BM25 + BERT scores.
        """
        effective_candidates = max(bm25_candidates, top_k * 2, 150)
        print(f"  [Serial] Stage 1: BM25 retrieving {effective_candidates} candidates...")
        bm25_results = self.bm25.search(query, top_k=effective_candidates)

        if not bm25_results:
            return []

        bm25_score_map = {r["doc_id"]: r["score"] for r in bm25_results}
        candidate_texts = {
            r["doc_id"]: r.get("full_text", r.get("text", ""))
            for r in bm25_results
        }

        encode_query = bert_query if bert_query else query
        print(f"  [Serial] Stage 2: BERT re-ranking {len(candidate_texts)} candidates...")

        query_vector = self.bert.encode_query(encode_query)

        bert_scores: Dict[str, float] = {}
        for doc_id, text in candidate_texts.items():
            doc_vector = self.bert.get_vector(doc_id)
            if doc_vector is not None:
                bert_scores[doc_id] = float(np.dot(query_vector, doc_vector))
            else:
                dv = self.bert.model.encode(text) if text else None
                bert_scores[doc_id] = float(np.dot(query_vector, dv)) if dv is not None else 0.0

        def _minmax(d: Dict[str, float]) -> Dict[str, float]:
            if not d:
                return {}
            lo, hi = min(d.values()), max(d.values())
            rng = hi - lo
            return {k: 1.0 for k in d} if rng == 0 else {k: (v - lo) / rng for k, v in d.items()}

        norm_bm25 = _minmax(bm25_score_map)
        norm_bert  = _minmax(bert_scores)
        total_w = bm25_weight + bert_weight
        w_bm25  = bm25_weight / total_w
        w_bert  = bert_weight  / total_w

        combined = sorted(
            [
                (doc_id, w_bm25 * norm_bm25.get(doc_id, 0.0) + w_bert * norm_bert.get(doc_id, 0.0), text)
                for doc_id, text in candidate_texts.items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for doc_id, score, text in combined[:top_k]:
            preview = text[:300] + ("..." if len(text) > 300 else "")
            results.append({
                "doc_id":    doc_id,
                "score":     round(score, 6),
                "text":      preview,
                "full_text": text,
                "method":    f"Hybrid Serial (BM25→BERT w={bm25_weight:.1f}/{bert_weight:.1f})",
            })

        print(f"  [Serial] Returning {len(results)} results")
        return results

    # ── Parallel ──────────────────────────────────────────────────────────────

    def _parallel_search(
        self,
        query: str,
        top_k: int,
        fusion: str,
        bm25_weight: float,
        bert_weight: float,
        bert_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        retrieve_k = max(top_k * 5, 200)

        print(f"  [Parallel] Running BM25 (retrieve_k={retrieve_k})...")
        bm25_results = self.bm25.search(query, top_k=retrieve_k)

        encode_query = bert_query if bert_query else query
        print(f"  [Parallel] Running BERT...")
        bert_results = self.bert.search(encode_query, top_k=retrieve_k)

        if not bm25_results and not bert_results:
            return []

        bm25_ranks  = {r["doc_id"]: i + 1 for i, r in enumerate(bm25_results)}
        bert_ranks  = {r["doc_id"]: i + 1 for i, r in enumerate(bert_results)}
        bm25_scores = {r["doc_id"]: r["score"] for r in bm25_results}
        bert_scores = {r["doc_id"]: r["score"] for r in bert_results}
        all_doc_ids = set(bm25_ranks) | set(bert_ranks)

        if fusion == "rrf":
            fused = self._fuse_rrf(all_doc_ids, bm25_ranks, bert_ranks,
                                   bm25_weight=bm25_weight, bert_weight=bert_weight)
        elif fusion == "linear":
            fused = self._fuse_linear(all_doc_ids, bm25_scores, bert_scores,
                                      bm25_weight, bert_weight)
        else:
            raise ValueError(f"Unknown fusion '{fusion}'. Use 'rrf' or 'linear'.")

        fused.sort(key=lambda x: x[1], reverse=True)

        doc_text_map = {r["doc_id"]: r.get("full_text", r.get("text", "")) for r in bm25_results}
        for r in bert_results:
            if r["doc_id"] not in doc_text_map:
                doc_text_map[r["doc_id"]] = r.get("full_text", r.get("text", ""))

        results = []
        for doc_id, score in fused[:top_k]:
            text    = doc_text_map.get(doc_id, "")
            preview = text[:300] + ("..." if len(text) > 300 else "")
            results.append({
                "doc_id":    doc_id,
                "score":     round(score, 6),
                "text":      preview,
                "full_text": text,
                "method":    f"Hybrid Parallel {fusion.upper()} (BM25 + BERT)",
                "bm25_rank": bm25_ranks.get(doc_id, "-"),
                "bert_rank": bert_ranks.get(doc_id, "-"),
            })

        print(f"  [Parallel] Returning {len(results)} results (fusion={fusion})")
        return results

    # ── Fusion methods ────────────────────────────────────────────────────────

    def _fuse_rrf(
        self,
        doc_ids: set,
        bm25_ranks: Dict[str, int],
        bert_ranks: Dict[str, int],
        k: int = RRF_K,
        bm25_weight: float = 1.0,
        bert_weight: float = 1.0,
    ) -> List[Tuple[str, float]]:
        max_rank = max(
            max(bm25_ranks.values(), default=0),
            max(bert_ranks.values(), default=0),
        ) + 1
        return [
            (
                doc_id,
                (bm25_weight / (k + bm25_ranks.get(doc_id, max_rank))) +
                (bert_weight  / (k + bert_ranks.get(doc_id,  max_rank))),
            )
            for doc_id in doc_ids
        ]

    def _fuse_linear(
        self,
        doc_ids: set,
        bm25_scores: Dict[str, float],
        bert_scores: Dict[str, float],
        bm25_weight: float,
        bert_weight: float,
    ) -> List[Tuple[str, float]]:
        def minmax(scores):
            if not scores:
                return {}
            lo, hi = min(scores.values()), max(scores.values())
            rng = hi - lo
            return {k: 1.0 for k in scores} if rng == 0 else {k: (v - lo) / rng for k, v in scores.items()}

        n_bm25 = minmax(bm25_scores)
        n_bert  = minmax(bert_scores)
        return [
            (doc_id, bm25_weight * n_bm25.get(doc_id, 0.0) + bert_weight * n_bert.get(doc_id, 0.0))
            for doc_id in doc_ids
        ]

    # ── BM25 parameter passthrough ────────────────────────────────────────────

    @property
    def k1(self) -> float:
        return self.bm25.k1

    @k1.setter
    def k1(self, value: float):
        self.bm25.k1 = value

    @property
    def b(self) -> float:
        return self.bm25.b

    @b.setter
    def b(self, value: float):
        self.bm25.b = value

    def get_stats(self) -> Dict[str, Any]:
        return {
            "bm25_documents":    self.bm25.document_store.total_docs,
            "bm25_unique_terms": len(self.bm25.inverted_index),
            "bm25_avg_length":   self.bm25.document_store.avg_doc_length,
            "bert_documents":    self.bert.total_docs,
            "bert_model":        self.bert.model.model_name,
            "bert_vector_dim":   self.bert.model.dim,
            "bm25_k1":           self.bm25.k1,
            "bm25_b":            self.bm25.b,
        }
