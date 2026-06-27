# services/evaluation/evaluation_service.py
"""
Evaluation Service — MAP, P@10, nDCG@10, Recall@100

Uses ClinicalTrials (TREC PM 2017) as the TEST SET.
Requires pytrec-eval-terrier:
    pip install pytrec-eval-terrier
"""

import ir_datasets
import pytrec_eval
import pickle
import os
from typing import Dict, Callable, Any, Tuple, Optional


class EvaluationService:
    """
    Evaluates IR models using official qrels from ClinicalTrials (TREC PM 2017).
    """

    METRICS = {"map", "ndcg_cut_10", "P_10", "recall_100"}

    def __init__(self, indexed_doc_ids: set):
        """
        Args:
            indexed_doc_ids: set of doc_ids that are actually indexed.
                             Used to filter qrels to docs we can retrieve.
        """
        self.indexed_doc_ids = indexed_doc_ids
        self.queries: Dict[str, str] = {}
        self.qrels: Dict[str, Dict[str, int]] = {}
        self.eval_queries: Dict[str, str] = {}

        # Load ClinicalTrials data — wrapped in try/except so a missing
        # pickle file or ir_datasets download failure doesn't crash the service.
        try:
            self._load_eval_data_clinical()
        except Exception as e:
            print(f"⚠️  Could not load eval data: {e}")
            print("   Run  python scripts/process_docs.py  first to generate pickle files.")

    # ── Data Loading ──────────────────────────────────────────────────────────

    def _load_eval_data_clinical(self) -> None:
        """
        Load queries and qrels from ClinicalTrials (TREC PM 2017).
        Uses the pre-processed pickle files if available,
        otherwise falls back to ir_datasets.
        """
        print("[EvaluationService] Loading ClinicalTrials (TREC 2017) queries and qrels...")

        queries_path = 'data/processed/queries.pkl'
        qrels_path   = 'data/processed/qrels.pkl'

        # ── Fast path: use cached pickle files ────────────────────────────────
        if os.path.exists(queries_path) and os.path.exists(qrels_path):
            with open(queries_path, 'rb') as f:
                self.queries = pickle.load(f)

            with open(qrels_path, 'rb') as f:
                raw_qrels = pickle.load(f)

            self.qrels = {}
            skipped = 0
            # raw_qrels is a list of tuples: (query_id, doc_id, relevance)
            for qid, doc_id, relevance in raw_qrels:
                qid    = str(qid)
                doc_id = str(doc_id)
                if doc_id in self.indexed_doc_ids:
                    if qid not in self.qrels:
                        self.qrels[qid] = {}
                    self.qrels[qid][doc_id] = int(relevance)
                else:
                    skipped += 1

            print(f"  ✅ Loaded queries  → {queries_path}")
            print(f"  ✅ Loaded qrels    → {qrels_path}  ({skipped:,} skipped — doc not in index)")

        # ── Slow path: download / read via ir_datasets ────────────────────────
        else:
            print("  ⚠️  Pickle files not found. Falling back to ir_datasets …")
            dataset = ir_datasets.load('clinicaltrials/2017/trec-pm-2017')

            # Queries
            self.queries = {}
            for q in dataset.queries_iter():
                qid = str(q.query_id)
                if hasattr(q, 'text'):
                    text = q.text
                elif hasattr(q, 'default_text'):
                    text = q.default_text()
                else:
                    text = str(q)
                self.queries[qid] = text

            # Qrels
            self.qrels  = {}
            skipped     = 0
            for qrel in dataset.qrels_iter():
                qid    = str(qrel.query_id)
                doc_id = str(qrel.doc_id)
                if doc_id in self.indexed_doc_ids:
                    if qid not in self.qrels:
                        self.qrels[qid] = {}
                    self.qrels[qid][doc_id] = int(qrel.relevance)
                else:
                    skipped += 1

            print(
                f"  ✅ Loaded via ir_datasets: "
                f"{len(self.queries)} queries, {len(self.qrels)} with qrels "
                f"({skipped:,} skipped)"
            )

        # ── Keep only queries that have ≥1 relevant doc in our index ──────────
        self.eval_queries = {
            qid: text
            for qid, text in self.queries.items()
            if qid in self.qrels and len(self.qrels[qid]) > 0
        }
        print(f"  ✅ Evaluable queries: {len(self.eval_queries):,}")

        if len(self.eval_queries) == 0:
            print(
                "  ⚠️  WARNING: 0 evaluable queries — "
                "your indexed docs may not overlap with ClinicalTrials relevant docs. "
                "Check your index and qrels."
            )

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        search_fn: Callable[[str, int], list],
        top_k: int = 100,
        max_queries: Optional[int] = None,
        label: str = "Model",
        verbose: bool = True,
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Run evaluation against all evaluable queries.

        Args:
            search_fn:   function(query_text, top_k) → list of {'doc_id', 'score', …}
            top_k:       number of results to retrieve per query
            max_queries: limit queries for faster testing (None = all)
            label:       model name for logging
            verbose:     print progress and per-model results

        Returns:
            (aggregated_metrics, per_query_metrics)
            aggregated_metrics keys: 'map', 'ndcg_cut_10', 'P_10', 'recall_100', 'recall'
        """
        queries_to_run = (
            dict(list(self.eval_queries.items())[:max_queries])
            if max_queries
            else self.eval_queries
        )

        if not queries_to_run:
            print(f"⚠️  No evaluable queries for {label}.")
            return {}, {}

        if verbose:
            print(f"\n[EvaluationService] Evaluating '{label}' on {len(queries_to_run):,} queries …")

        # Build run dict:  query_id → {doc_id: score}
        run: Dict[str, Dict[str, float]] = {}
        for i, (query_id, query_text) in enumerate(queries_to_run.items(), 1):
            results           = search_fn(query_text, top_k)
            run[query_id]     = {r["doc_id"]: float(r["score"]) for r in results}
            if verbose and i % 10 == 0:
                print(f"  Queried {i}/{len(queries_to_run)}")

        # Filter qrels to only the queries we actually ran
        eval_qrels = {qid: self.qrels[qid] for qid in run if qid in self.qrels}

        if not eval_qrels:
            print(f"⚠️  No qrels found for the evaluated queries. Skipping '{label}'.")
            return {}, {}

        # Official TREC evaluation via pytrec_eval
        evaluator = pytrec_eval.RelevanceEvaluator(eval_qrels, self.METRICS)
        per_query = evaluator.evaluate(run)

        # Aggregate: mean over all queries
        aggregated: Dict[str, float] = {}
        for metric in self.METRICS:
            values             = [v[metric] for v in per_query.values() if metric in v]
            aggregated[metric] = round(sum(values) / len(values), 4) if values else 0.0

        # Alias recall_100 → recall for backward compatibility
        if "recall_100" in aggregated:
            aggregated["recall"] = aggregated["recall_100"]

        if verbose:
            print(f"\n  ── {label} ──")
            print(f"  MAP        : {aggregated.get('map',          0):.4f}")
            print(f"  nDCG@10    : {aggregated.get('ndcg_cut_10',  0):.4f}")
            print(f"  P@10       : {aggregated.get('P_10',         0):.4f}")
            print(f"  Recall@100 : {aggregated.get('recall_100',   0):.4f}")

        return aggregated, per_query

    # ── Convenience helpers ───────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the loaded evaluation data."""
        return {
            "total_queries":     len(self.queries),
            "evaluable_queries": len(self.eval_queries),
            "indexed_docs":      len(self.indexed_doc_ids),
            "qrels_coverage":    len(self.qrels),
        }

    def get_queries(self) -> Dict[str, str]:
        """Return all loaded queries as {query_id: text}."""
        return self.queries

    def get_qrels(self) -> Dict[str, Dict[str, int]]:
        """Return all loaded qrels as {query_id: {doc_id: relevance}}."""
        return self.qrels