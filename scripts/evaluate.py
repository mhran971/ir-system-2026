# scripts/evaluate.py
"""
Evaluation Script — compares all IR models on MS MARCO dev/small.

Run after building all indexes:
    python scripts/evaluate.py

What it does:
    1. Loads all services (BM25, VSM, BERT, Hybrid)
    2. Loads EvaluationService with dev/small qrels (TEST SET)
    3. Runs evaluation for each model
    4. Prints comparison table: MAP | nDCG@10 | P@10 | Recall@100
    5. Saves results to data/evaluation/results.json

Metrics:
    MAP           — Mean Average Precision (main IR metric)
    nDCG@10       — Normalized Discounted Cumulative Gain at 10
    P@10          — Precision at 10
    Recall@100    — Recall within top-100 results
"""

import sys
import os
import json
import time

sys.stdout.reconfigure(encoding="utf-8")
os.environ["PYTHONUTF8"] = "1"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.retrieval.bm25_search_service import BM25SearchService
from services.retrieval.vsm_search_service import VSMSearchService
from services.ranking.embeddings.bert_search_service import BERTSearchService
from services.ranking.hybrid.hybrid_search_service import HybridSearchService
from services.evaluation.evaluation_service import EvaluationService


# ── Config ────────────────────────────────────────────────────────────────────

TOP_K       = 100   # retrieve 100 results per query (needed for Recall@100)
MAX_QUERIES = 200   # set to None to run all evaluable queries (slower)
                    # 200 is enough for a meaningful comparison


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_table(results: dict) -> None:
    """Print a formatted comparison table."""
    header = f"{'Model':<30} {'MAP':>8} {'nDCG@10':>10} {'P@10':>8} {'Recall@100':>12}"
    sep    = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for model_name, metrics in results.items():
        print(
            f"{model_name:<30} "
            f"{metrics.get('map', 0):>8.4f} "
            f"{metrics.get('ndcg_cut_10', 0):>10.4f} "
            f"{metrics.get('P_10', 0):>8.4f} "
            f"{metrics.get('recall_100', 0):>12.4f}"
        )
    print(sep)

    # Best model per metric
    print("\n🏆 Best per metric:")
    for metric, label in [("map","MAP"), ("ndcg_cut_10","nDCG@10"),
                           ("P_10","P@10"), ("recall_100","Recall@100")]:
        best = max(results, key=lambda m: results[m].get(metric, 0))
        print(f"   {label:>12} → {best}  ({results[best].get(metric, 0):.4f})")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("📊 IR System 2026 — Model Evaluation")
    print(f"   Test set : msmarco-passage/dev/small")
    print(f"   Top-K    : {TOP_K}")
    print(f"   Queries  : {MAX_QUERIES or 'all evaluable'}")
    print("=" * 60)

    # ── Load services ─────────────────────────────────────────────────────────
    print("\n[1/6] Loading BM25 service...")
    bm25_svc = BM25SearchService(k1=1.5, b=0.75)

    print("\n[2/6] Loading VSM service...")
    vsm_svc = VSMSearchService()

    print("\n[3/6] Loading BERT service...")
    bert_svc = BERTSearchService(model_key="fast")

    print("\n[4/6] Loading Hybrid service...")
    hybrid_svc = HybridSearchService(
        bm25_service=bm25_svc,
        bert_service=bert_svc,
    )

    # ── Build indexed_doc_ids from DocumentStore ───────────────────────────────
    # EvaluationService needs to know which doc_ids are indexed
    # to filter qrels to only docs we can retrieve
    print("\n[5/6] Collecting indexed doc IDs...")
    indexed_doc_ids = set(bm25_svc.document_store.get_all_documents().keys())
    print(f"  Indexed docs: {len(indexed_doc_ids):,}")

    # ── Init EvaluationService (loads dev/small qrels) ────────────────────────
    print("\n[6/6] Initialising EvaluationService (dev/small TEST SET)...")
    eval_svc = EvaluationService(indexed_doc_ids=indexed_doc_ids)
    stats = eval_svc.get_stats()
    print(f"  Total dev/small queries : {stats['total_queries']:,}")
    print(f"  Evaluable queries       : {stats['evaluable_queries']:,}")

    if stats["evaluable_queries"] == 0:
        print("\n⚠️  No evaluable queries — cannot continue.")
        print("   Make sure you indexed enough documents so some dev/small")
        print("   relevant docs are in your index.")
        return

    # ── Run evaluation ────────────────────────────────────────────────────────
    all_results = {}

    # 1. BM25
    t = time.time()
    agg, _ = eval_svc.evaluate(
        search_fn=lambda q, k: bm25_svc.search(q, top_k=k),
        top_k=TOP_K, max_queries=MAX_QUERIES, label="BM25 (rank_bm25)"
    )
    all_results["BM25"] = {**agg, "time_s": round(time.time() - t, 1)}

    # 2. VSM TF-IDF
    t = time.time()
    agg, _ = eval_svc.evaluate(
        search_fn=lambda q, k: vsm_svc.search(q, top_k=k),
        top_k=TOP_K, max_queries=MAX_QUERIES, label="VSM TF-IDF (sklearn)"
    )
    all_results["VSM TF-IDF"] = {**agg, "time_s": round(time.time() - t, 1)}

    # 3. BERT
    t = time.time()
    agg, _ = eval_svc.evaluate(
        search_fn=lambda q, k: bert_svc.search(q, top_k=k),
        top_k=TOP_K, max_queries=MAX_QUERIES, label="BERT (sentence-transformers)"
    )
    all_results["BERT"] = {**agg, "time_s": round(time.time() - t, 1)}

    # 4. Hybrid Serial
    t = time.time()
    agg, _ = eval_svc.evaluate(
        search_fn=lambda q, k: hybrid_svc.search(
            q, mode="serial", top_k=k, bm25_candidates=100
        ),
        top_k=TOP_K, max_queries=MAX_QUERIES, label="Hybrid Serial"
    )
    all_results["Hybrid Serial"] = {**agg, "time_s": round(time.time() - t, 1)}

    # 5. Hybrid Parallel RRF
    t = time.time()
    agg, _ = eval_svc.evaluate(
        search_fn=lambda q, k: hybrid_svc.search(
            q, mode="parallel", fusion="rrf", top_k=k
        ),
        top_k=TOP_K, max_queries=MAX_QUERIES, label="Hybrid Parallel RRF"
    )
    all_results["Hybrid Parallel RRF"] = {**agg, "time_s": round(time.time() - t, 1)}

    # 6. Hybrid Parallel Linear
    t = time.time()
    agg, _ = eval_svc.evaluate(
        search_fn=lambda q, k: hybrid_svc.search(
            q, mode="parallel", fusion="linear",
            top_k=k, bm25_weight=0.4, bert_weight=0.6
        ),
        top_k=TOP_K, max_queries=MAX_QUERIES, label="Hybrid Parallel Linear"
    )
    all_results["Hybrid Parallel Linear"] = {**agg, "time_s": round(time.time() - t, 1)}

    # ── Print comparison table ─────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("📊 MODEL COMPARISON — msmarco-passage/dev/small")
    print("=" * 60)
    print_table(all_results)

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs("data/evaluation", exist_ok=True)
    out_path = "data/evaluation/results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_set":      "msmarco-passage/dev/small",
                "top_k":         TOP_K,
                "max_queries":   MAX_QUERIES,
                "indexed_docs":  len(indexed_doc_ids),
                "eval_queries":  stats["evaluable_queries"],
                "models":        all_results,
            },
            f, indent=2, ensure_ascii=False
        )
    print(f"\n✅ Results saved → {out_path}")
    print("\nNext: use these numbers in your report and UI evaluation tab.")


if __name__ == "__main__":
    main()