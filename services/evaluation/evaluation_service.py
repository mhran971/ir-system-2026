# services/evaluation/evaluation_service.py
"""
Evaluation Service — MAP, P@10, nDCG@10, Recall@100

Uses MS MARCO dev/small as the TEST SET (NOT train queries).
Requires pytrec-eval-terrier:
    pip install pytrec-eval-terrier
"""

import ir_datasets
import pytrec_eval
from typing import Dict, Callable, Any, Tuple


class EvaluationService:
    """
    Evaluates IR models using official MS MARCO dev/small qrels.

    Why dev/small and NOT train?
    - train queries have NO official qrels available for proper evaluation
    - dev/small has 6,980 queries with official relevance judgments (qrels)
    - Evaluating on train queries = measuring what you indexed, not retrieval quality
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

        self._load_eval_data()

    # ── Data Loading ──────────────────────────────────────────────────────────

    def _load_eval_data(self) -> None:
        """
        Load queries and qrels from msmarco-passage/dev/small.
        Filters out queries whose relevant docs are not in our index.
        """
        print("[EvaluationService] Loading dev/small queries and qrels...")

        # ✅ TEST SET: msmarco-passage/dev/small
        #    NOT msmarco-passage/train (those have no public qrels)
        dataset = ir_datasets.load("msmarco-passage/dev/small")

        # Load all queries
        self.queries = {
            q.query_id: q.text
            for q in dataset.queries_iter()
        }
        print(f"  Loaded {len(self.queries):,} queries from dev/small")

        # Load qrels — filter to docs we have indexed
        # (our index may only cover a subset of the full 8.8M corpus)
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
            aggregated_metrics keys: 'map', 'ndcg_cut_10', 'P_10', 'recall_100'
        """
        queries_to_run = dict(list(self.eval_queries.items())[:max_queries]) \
            if max_queries else self.eval_queries

        print(f"\n[EvaluationService] Evaluating {label} on {len(queries_to_run):,} queries...")

        # Build run dict: query_id → {doc_id: score}
        run: Dict[str, Dict[str, float]] = {}
        for i, (query_id, query_text) in enumerate(queries_to_run.items(), 1):
            results = search_fn(query_text, top_k)
            run[query_id] = {r["doc_id"]: float(r["score"]) for r in results}
            if i % 100 == 0:
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