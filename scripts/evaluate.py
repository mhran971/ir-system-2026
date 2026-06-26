# scripts/evaluate.py
"""
Evaluation Script — compares all IR models on ClinicalTrials 2017 (TREC 2017).
Evaluates each model twice: Baseline (before enhancements) and + Refinement (after enhancements).

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
from services.query_processing.query_refiner import QueryRefiner

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
    print("   Evaluating: Baseline vs Enhanced (+ Refinement)")
    print(f"   Top-K    : {TOP_K}")
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

    # ── Build indexed_doc_ids ────────────────────────────────────────────────
    print("\n[5/6] Collecting indexed doc IDs...")
    indexed_doc_ids = set(bm25_svc.document_store.get_all_documents().keys())
    print(f"  Indexed docs: {len(indexed_doc_ids):,}")

    # ── Init QueryRefiner ─────────────────────────────────────────────────────
    print("\n[6/6] Initializing QueryRefiner...")
    known_terms = set(bm25_svc.inverted_index.doc_frequency.keys())
    term_freqs = dict(bm25_svc.inverted_index.doc_frequency)
    total_docs = bm25_svc.document_store.total_docs
    refiner = QueryRefiner(
        known_terms=known_terms,
        term_frequencies=term_freqs,
        total_docs=total_docs
    )

    # ── Init EvaluationService (TREC 2017 qrels) ─────────────────────────────
    print("\n[7/7] Initialising EvaluationService (TREC 2017)...")
    eval_svc = EvaluationService(indexed_doc_ids=indexed_doc_ids)
    
    # Load ClinicalTrials dataset
    eval_svc._load_eval_data_clinical()

    stats = eval_svc.get_stats()
    print(f"  Total queries    : {stats['total_queries']:,}")
    print(f"  Evaluable queries: {stats['evaluable_queries']:,}")

    if stats["evaluable_queries"] == 0:
        print("\n⚠️  No evaluable queries — cannot continue.")
        return

    # ── Helper search function with refinement ──────────────────────────────
    def refined_search(search_fn, q, k):
        # 1. PRF Docs using BM25
        prf_docs = bm25_svc.search(q, top_k=5)
        # 2. Refine query
        refined = refiner.refine(
            query=q,
            apply_spelling=True,
            apply_synonyms=True,
            apply_prf=True,
            apply_history=False,  # disabled for batch evaluation consistency
            top_docs=prf_docs,
            num_prf_terms=3,
            synonym_limit=1
        )
        refined_q = refined["expanded_query"]
        # 3. Execute search on refined query
        return search_fn(refined_q, k)

    # ── Run evaluation ────────────────────────────────────────────────────────
    all_results = {}

    # 1. BM25
    agg_base, _ = eval_svc.evaluate(
        search_fn=lambda q, k: bm25_svc.search(q, top_k=k),
        top_k=TOP_K, max_queries=MAX_QUERIES, label="BM25 (Baseline)"
    )
    all_results["BM25 (Baseline)"] = agg_base

    agg_enh, _ = eval_svc.evaluate(
        search_fn=lambda q, k: refined_search(lambda rq, rk: bm25_svc.search(rq, top_k=rk), q, k),
        top_k=TOP_K, max_queries=MAX_QUERIES, label="BM25 (+ Refinement)"
    )
    all_results["BM25 (+ Refinement)"] = agg_enh

    # 2. VSM TF-IDF
    agg_base, _ = eval_svc.evaluate(
        search_fn=lambda q, k: vsm_svc.search(q, top_k=k),
        top_k=TOP_K, max_queries=MAX_QUERIES, label="VSM TF-IDF (Baseline)"
    )
    all_results["VSM TF-IDF (Baseline)"] = agg_base

    agg_enh, _ = eval_svc.evaluate(
        search_fn=lambda q, k: refined_search(lambda rq, rk: vsm_svc.search(rq, top_k=rk), q, k),
        top_k=TOP_K, max_queries=MAX_QUERIES, label="VSM TF-IDF (+ Refinement)"
    )
    all_results["VSM TF-IDF (+ Refinement)"] = agg_enh

    # 3. BERT
    agg_base, _ = eval_svc.evaluate(
        search_fn=lambda q, k: bert_svc.search(q, top_k=k),
        top_k=TOP_K, max_queries=MAX_QUERIES, label="BERT (Baseline)"
    )
    all_results["BERT (Baseline)"] = agg_base

    agg_enh, _ = eval_svc.evaluate(
        search_fn=lambda q, k: refined_search(lambda rq, rk: bert_svc.search(rq, top_k=rk), q, k),
        top_k=TOP_K, max_queries=MAX_QUERIES, label="BERT (+ Refinement)"
    )
    all_results["BERT (+ Refinement)"] = agg_enh

    # 4. Hybrid Serial
    agg_base, _ = eval_svc.evaluate(
        search_fn=lambda q, k: hybrid_svc.search(
            q, mode="serial", top_k=k, bm25_candidates=100
        ),
        top_k=TOP_K, max_queries=MAX_QUERIES, label="Hybrid Serial (Baseline)"
    )
    all_results["Hybrid Serial (Baseline)"] = agg_base

    agg_enh, _ = eval_svc.evaluate(
        search_fn=lambda q, k: refined_search(
            lambda rq, rk: hybrid_svc.search(rq, mode="serial", top_k=rk, bm25_candidates=100), q, k
        ),
        top_k=TOP_K, max_queries=MAX_QUERIES, label="Hybrid Serial (+ Refinement)"
    )
    all_results["Hybrid Serial (+ Refinement)"] = agg_enh

    # 5. Hybrid Parallel RRF
    agg_base, _ = eval_svc.evaluate(
        search_fn=lambda q, k: hybrid_svc.search(
            q, mode="parallel", fusion="rrf", top_k=k
        ),
        top_k=TOP_K, max_queries=MAX_QUERIES, label="Hybrid Parallel RRF (Baseline)"
    )
    all_results["Hybrid Parallel RRF (Baseline)"] = agg_base

    agg_enh, _ = eval_svc.evaluate(
        search_fn=lambda q, k: refined_search(
            lambda rq, rk: hybrid_svc.search(rq, mode="parallel", fusion="rrf", top_k=rk), q, k
        ),
        top_k=TOP_K, max_queries=MAX_QUERIES, label="Hybrid Parallel RRF (+ Refinement)"
    )
    all_results["Hybrid Parallel RRF (+ Refinement)"] = agg_enh

    # ── Print comparison table ───────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("📊 MODEL COMPARISON — ClinicalTrials 2017 (TREC 2017)")
    print("=" * 60)
    print_table(all_results)

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs("data/evaluation", exist_ok=True)
    out_path = "data/evaluation/results_clinical.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_set":      "clinicaltrials/2017/trec-pm-2017",
                "top_k":         TOP_K,
                "indexed_docs":  len(indexed_doc_ids),
                "eval_queries":  stats["evaluable_queries"],
                "models":        all_results,
            },
            f, indent=2, ensure_ascii=False
        )
    print(f"\n✅ Results saved → {out_path}")


if __name__ == "__main__":
    main()
