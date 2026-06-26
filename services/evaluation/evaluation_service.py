# services/evaluation/evaluation_service.py
"""
Evaluation Service — MAP, P@10, nDCG@10, Recall@100

Uses MS MARCO dev/small or ClinicalTrials (TREC PM 2017) as the TEST SET.
Requires pytrec-eval-terrier:
    pip install pytrec-eval-terrier
"""

import ir_datasets
import pytrec_eval
import pickle
import os
from typing import Dict, Callable, Any, Tuple


class EvaluationService:
    """
    Evaluates IR models using official qrels.
    Supports both MS MARCO dev/small and ClinicalTrials (TREC PM 2017).
    """

    METRICS = {"map", "ndcg_cut_10", "P_10", "recall_100"}

    def __init__(self, indexed_doc_ids: set):
        """
        Args:
            indexed_doc_ids: set of doc_ids that are actually indexed.
                             Used to filter qrels to docs we can retrieve.
        """
        self.indexed_doc_ids = indexed_doc_ids
        self.queries: Dict[str, str] = {}        # query_id → query text
        self.qrels:   Dict[str, Dict[str, int]] = {}  # query_id → {doc_id: relevance}
        self.eval_queries: Dict[str, str] = {}   # filtered: only queries with indexed relevant docs

    # ── Data Loading ──────────────────────────────────────────────────────────

    def _load_eval_data(self) -> None:
        """
        Load queries and qrels from msmarco-passage/dev/small.
        Filters out queries whose relevant docs are not in our index.
        """
        print("[EvaluationService] Loading dev/small queries and qrels...")

        dataset = ir_datasets.load("msmarco-passage/dev/small")

        # Load all queries
        self.queries = {
            q.query_id: q.text
            for q in dataset.queries_iter()
        }
        print(f"  Loaded {len(self.queries):,} queries from dev/small")

        # Load qrels — filter to docs we have indexed
        skipped = 0
        for qrel in dataset.qrels_iter():
            if qrel.doc_id in self.indexed_doc_ids:
                if qrel.query_id not in self.qrels:
                    self.qrels[qrel.query_id] = {}
                self.qrels[qrel.query_id][qrel.doc_id] = int(qrel.relevance)
            else:
                skipped += 1

        print(f"  Loaded qrels for {len(self.qrels):,} queries "
              f"({skipped:,} qrels skipped — relevant docs not in index)")

        # Only evaluate queries that have ≥1 relevant doc in our index
        self.eval_queries = {
            qid: text
            for qid, text in self.queries.items()
            if qid in self.qrels and len(self.qrels[qid]) > 0
        }
        print(f"  Evaluable queries (have indexed relevant docs): {len(self.eval_queries):,}")

        if len(self.eval_queries) == 0:
            print("  ⚠️  WARNING: 0 evaluable queries — your indexed docs may not overlap "
                  "with dev/small relevant docs. Index more documents or check doc IDs.")

    def _load_eval_data_clinical(self) -> None:
        """
        Load queries and qrels from ClinicalTrials (TREC PM 2017).
        Uses the pre-processed pickle files if available, otherwise falls back to ir_datasets.
        """
        print("[EvaluationService] Loading ClinicalTrials (TREC 2017) queries and qrels...")

        queries_path = 'data/processed/queries.pkl'
        qrels_path = 'data/processed/qrels.pkl'

        if os.path.exists(queries_path) and os.path.exists(qrels_path):
            with open(queries_path, 'rb') as f:
                self.queries = pickle.load(f)

            with open(qrels_path, 'rb') as f:
                raw_qrels = pickle.load(f)

            self.qrels = {}
            skipped = 0
            # raw_qrels is list of tuples: (query_id, doc_id, relevance)
            for qid, doc_id, relevance in raw_qrels:
                qid = str(qid)
                doc_id = str(doc_id)
                if doc_id in self.indexed_doc_ids:
                    if qid not in self.qrels:
                        self.qrels[qid] = {}
                    self.qrels[qid][doc_id] = int(relevance)
                else:
                    skipped += 1
            print(f"  Loaded queries from {queries_path}")
            print(f"  Loaded qrels from {qrels_path} ({skipped:,} skipped)")
        else:
            print("  Pickled files not found. Loading via ir_datasets...")
            dataset = ir_datasets.load('clinicaltrials/2017/trec-pm-2017')
            self.queries = {
                str(q.query_id): (q.text if hasattr(q, 'text') else q.default_text())
                for q in dataset.queries_iter()
            }
            self.qrels = {}
            skipped = 0
            for qrel in dataset.qrels_iter():
                qid = str(qrel.query_id)
                doc_id = str(qrel.doc_id)
                if doc_id in self.indexed_doc_ids:
                    if qid not in self.qrels:
                        self.qrels[qid] = {}
                    self.qrels[qid][doc_id] = int(qrel.relevance)
                else:
                    skipped += 1
            print(f"  Loaded via ir_datasets: {len(self.queries)} queries, {len(self.qrels)} qrels")

        # Only evaluate queries that have ≥1 relevant doc in our index
        self.eval_queries = {
            qid: text
            for qid, text in self.queries.items()
            if qid in self.qrels and len(self.qrels[qid]) > 0
        }
        print(f"  Evaluable queries (have indexed relevant docs): {len(self.eval_queries):,}")

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        search_fn: Callable[[str, int], list],
        top_k: int = 100,
        max_queries: int = None,
        label: str = "Model",
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Run evaluation against all evaluable queries.

        Args:
            search_fn:   function(query_text, top_k) → list of {'doc_id', 'score', ...}
            top_k:       number of results to retrieve per query
            max_queries: limit queries for faster testing (None = all)
            label:       model name for logging

        Returns:
            (aggregated_metrics, per_query_metrics)
            aggregated_metrics keys: 'map', 'ndcg_cut_10', 'P_10', 'recall_100', 'recall'
        """
        queries_to_run = dict(list(self.eval_queries.items())[:max_queries]) \
            if max_queries else self.eval_queries

        print(f"\n[EvaluationService] Evaluating {label} on {len(queries_to_run):,} queries...")

        # Build run dict: query_id → {doc_id: score}
        run: Dict[str, Dict[str, float]] = {}
        for i, (query_id, query_text) in enumerate(queries_to_run.items(), 1):
            results = search_fn(query_text, top_k)
            run[query_id] = {r["doc_id"]: float(r["score"]) for r in results}
            if i % 10 == 0:
                print(f"  Queried {i}/{len(queries_to_run)}")

        # Filter qrels to only evaluated queries
        eval_qrels = {qid: self.qrels[qid] for qid in run if qid in self.qrels}

        # ✅ pytrec_eval — official TREC evaluation
        evaluator = pytrec_eval.RelevanceEvaluator(eval_qrels, self.METRICS)
        per_query = evaluator.evaluate(run)

        # Aggregate: mean across all queries
        aggregated: Dict[str, float] = {}
        for metric in self.METRICS:
            values = [v[metric] for v in per_query.values() if metric in v]
            aggregated[metric] = round(sum(values) / len(values), 4) if values else 0.0

        # Alias recall_100 to recall for printing/compatibility
        if "recall_100" in aggregated:
            aggregated["recall"] = aggregated["recall_100"]

        print(f"\n  ── {label} Results ──")
        print(f"  MAP           : {aggregated.get('map', 0):.4f}")
        print(f"  nDCG@10       : {aggregated.get('ndcg_cut_10', 0):.4f}")
        print(f"  P@10          : {aggregated.get('P_10', 0):.4f}")
        print(f"  Recall@100    : {aggregated.get('recall_100', 0):.4f}")

        return aggregated, per_query

    # ── Convenience ───────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_queries":     len(self.queries),
            "evaluable_queries": len(self.eval_queries),
            "indexed_docs":      len(self.indexed_doc_ids),
            "qrels_coverage":    len(self.qrels),
        }
