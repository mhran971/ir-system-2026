# scripts/evaluate.py
"""
Evaluation Script — compares all IR models on ClinicalTrials 2017 (TREC 2017).
Evaluates each model ONLY on Baseline (no Query Refinement).

Run after building all indexes:
    python scripts/evaluate.py
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

TOP_K       = 100   # retrieve 100 results per query
MAX_QUERIES = None  # None = all queries (30 queries in TREC 2017)


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_table(results: dict) -> None:
    """Print a formatted comparison table."""
    header = f"{'Model':<40} {'MAP':>8} {'nDCG@10':>10} {'P@10':>8} {'Recall':>10}"
    sep    = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for model_name, metrics in results.items():
        print(
            f"{model_name:<40} "
            f"{metrics.get('map', 0):>8.4f} "
            f"{metrics.get('ndcg_cut_10', 0):>10.4f} "
            f"{metrics.get('P_10', 0):>8.4f} "
            f"{metrics.get('recall', 0):>10.4f}"
        )
    print(sep)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("📊 IR System 2026 — Model Evaluation (TREC PM 2017)")
    print("   Evaluating: Baseline models ONLY (No Query Refinement)")
    print(f"   Top-K    : {TOP_K}")
    print("=" * 60)

    # ── Load services ─────────────────────────────────────────────────────────
    print("\n[1/5] Loading BM25 service...")
    bm25_svc = BM25SearchService(k1=1.5, b=0.75)

    print("\n[2/5] Loading VSM service...")
    vsm_svc = VSMSearchService()

    print("\n[3/5] Loading BERT service...")
    model_key = os.getenv("BERT_MODEL_KEY", "fast")
    bert_svc = BERTSearchService(model_key=model_key)

    print("\n[4/5] Loading Hybrid service...")
    hybrid_svc = HybridSearchService(
        bm25_service=bm25_svc,
        bert_service=bert_svc,
    )

    # ── Build indexed_doc_ids ────────────────────────────────────────────────
    print("\n[5/5] Collecting indexed doc IDs...")
    indexed_doc_ids = set(bm25_svc.document_store.get_all_documents().keys())
    print(f"  Indexed docs: {len(indexed_doc_ids):,}")

    # ── Init EvaluationService (TREC 2017 qrels) ─────────────────────────────
    print("\n[6/6] Initialising EvaluationService (TREC 2017)...")
    eval_svc = EvaluationService(indexed_doc_ids=indexed_doc_ids)
    
    # Load ClinicalTrials dataset
    eval_svc._load_eval_data_clinical()

    stats = eval_svc.get_stats()
    print(f"  Total queries    : {stats['total_queries']:,}")
    print(f"  Evaluable queries: {stats['evaluable_queries']:,}")

    if stats["evaluable_queries"] == 0:
        print("\n⚠️  No evaluable queries — cannot continue.")
        return

    # ── Run evaluation ────────────────────────────────────────────────────────
    all_results = {}

    # 1. BM25
    print("\n📊 Evaluating BM25...")
    agg, _ = eval_svc.evaluate(
        search_fn=lambda q, k: bm25_svc.search(q, top_k=k),
        top_k=TOP_K, max_queries=MAX_QUERIES, label="BM25"
    )
    all_results["BM25"] = agg

    # 2. VSM TF-IDF
    print("\n📊 Evaluating VSM TF-IDF...")
    agg, _ = eval_svc.evaluate(
        search_fn=lambda q, k: vsm_svc.search(q, top_k=k),
        top_k=TOP_K, max_queries=MAX_QUERIES, label="VSM TF-IDF"
    )
    all_results["VSM TF-IDF"] = agg

    # 3. BERT
    print("\n📊 Evaluating BERT...")
    agg, _ = eval_svc.evaluate(
        search_fn=lambda q, k: bert_svc.search(q, top_k=k),
        top_k=TOP_K, max_queries=MAX_QUERIES, label="BERT"
    )
    all_results["BERT"] = agg

    # 4. Hybrid Serial
    print("\n📊 Evaluating Hybrid Serial...")
    agg, _ = eval_svc.evaluate(
        search_fn=lambda q, k: hybrid_svc.search(
            q, mode="serial", top_k=k, bm25_candidates=300,
            bm25_weight=1.5, bert_weight=0.5,
        ),
        top_k=TOP_K, max_queries=MAX_QUERIES, label="Hybrid Serial"
    )
    all_results["Hybrid Serial"] = agg

    # 5. Hybrid Parallel RRF
    print("\n📊 Evaluating Hybrid Parallel RRF...")
    agg, _ = eval_svc.evaluate(
        search_fn=lambda q, k: hybrid_svc.search(
            q, mode="parallel", fusion="rrf", top_k=k,
            bm25_weight=1.5, bert_weight=0.5,
        ),
        top_k=TOP_K, max_queries=MAX_QUERIES, label="Hybrid Parallel RRF"
    )
    all_results["Hybrid Parallel RRF"] = agg

    # ── Print results ────────────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("📊 MODEL COMPARISON — ClinicalTrials 2017 (TREC 2017)")
    print("   (Baseline models — NO Query Refinement)")
    print("=" * 60)
    print_table(all_results)

    # ── Identify the best model ──────────────────────────────────────────────
    print("\n🏆 BEST MODEL BY METRIC:")
    print("-" * 60)
    
    # Find best per metric
    metrics = ['map', 'ndcg_cut_10', 'P_10', 'recall']
    metric_names = {
        'map': 'MAP',
        'ndcg_cut_10': 'nDCG@10',
        'P_10': 'P@10',
        'recall': 'Recall@100'
    }
    
    for metric in metrics:
        best_model = max(all_results.items(), key=lambda x: x[1].get(metric, 0))
        print(f"  {metric_names[metric]:<12} → {best_model[0]:<25} ({best_model[1].get(metric, 0):.4f})")

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs("data/evaluation", exist_ok=True)
    out_path = "data/evaluation/results_clinical_baseline_only.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_set":      "clinicaltrials/2017/trec-pm-2017",
                "top_k":         TOP_K,
                "description":   "Baseline models ONLY — NO Query Refinement",
                "indexed_docs":  len(indexed_doc_ids),
                "eval_queries":  stats["evaluable_queries"],
                "models":        all_results,
                "best_by_metric": {
                    metric: {
                        "model": best_model[0],
                        "score": best_model[1].get(metric, 0)
                    }
                    for metric in metrics
                    for best_model in [max(all_results.items(), key=lambda x: x[1].get(metric, 0))]
                }
            },
            f, indent=2, ensure_ascii=False
        )
    print(f"\n✅ Results saved → {out_path}")


if __name__ == "__main__":
    main()