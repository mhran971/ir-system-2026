# ui/app.py
"""
IR System 2026 — Streamlit UI
Models: BM25 | VSM TF-IDF | BERT Embeddings | Hybrid Serial | Hybrid Parallel | Simple TF-IDF
"""
import streamlit as st
import sys
import os
import time
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.retrieval.search_service import SearchService
from services.retrieval.vsm_search_service import VSMSearchService
from services.retrieval.bm25_search_service import BM25SearchService
from services.ranking.embeddings.bert_search_service import BERTSearchService
from services.ranking.hybrid.hybrid_search_service import HybridSearchService
from services.query_processing.query_refiner import QueryRefiner
from services.clustering.clustering_service import ClusteringService

def identity_analyzer(doc):
    """Pass-through analyzer for pre-tokenized inputs in VSM model unpickling."""
    return doc

# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource
def get_simple_service():
    return SearchService()

@st.cache_resource
def get_vsm_service():
    return VSMSearchService()

@st.cache_resource
def get_bm25_service():
    return BM25SearchService(k1=1.5, b=0.75)

@st.cache_resource
def get_bert_service():
    return BERTSearchService(model_key="fast")

@st.cache_resource
def get_hybrid_service():
    return HybridSearchService(
        bm25_service=get_bm25_service(),
        bert_service=get_bert_service(),
    )


@st.cache_resource
def get_refiner():
    try:
        bm25 = get_bm25_service()

        known_terms = set(bm25.inverted_index.doc_frequency.keys())
        term_freqs = dict(bm25.inverted_index.doc_frequency)

        total_docs = bm25.document_store.total_docs

        print(f"📚 Loaded {len(known_terms)} known terms for QueryRefiner")

        return QueryRefiner(
            known_terms=known_terms,
            term_frequencies=term_freqs,
            total_docs=total_docs
        )

    except Exception as e:
        st.warning(
            f"Could not load index vocabulary: {e}. Falling back to generic."
        )

        return QueryRefiner()

@st.cache_resource
def get_clustering_service():
    return ClusteringService(bert_service=get_bert_service())

# ── App ───────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="IR System 2026", page_icon="🔍", layout="wide")
    st.title("🔍 Information Retrieval System 2026")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ Settings")

        model = st.selectbox(
            "📊 Retrieval Model",
            [
                "BM25 (Best Matching 25)",
                "Hybrid Serial  (BM25 → BERT rerank)",
                "Hybrid Parallel  (BM25 + BERT fused)",
                "VSM TF-IDF (Cosine Similarity)",
                "BERT Embeddings (Semantic Search)",
                "Simple TF-IDF (Baseline)",
            ],
        )

        # ── BM25 parameters ───────────────────────────────────────────
        show_bm25_params = any(x in model for x in ["BM25", "Hybrid"])
        if show_bm25_params:
            st.markdown("---")
            st.subheader("🎯 BM25 Parameters")
            col1, col2 = st.columns(2)
            with col1:
                k1 = st.slider("k1 (TF Saturation)", 0.5, 2.5, 1.5, 0.1)
            with col2:
                b = st.slider("b (Length Norm)", 0.0, 1.0, 0.75, 0.05)
            st.caption(f"Current: **k1 = {k1}**, **b = {b}**")

        # ── Hybrid-specific options ───────────────────────────────────
        if "Hybrid Parallel" in model:
            st.markdown("---")
            st.subheader("🔀 Fusion Settings")
            fusion = st.radio("Fusion method", ["rrf", "linear"],
                              format_func=lambda x: "RRF" if x == "rrf" else "Linear")
            if fusion == "linear":
                bm25_w = st.slider("BM25 weight", 0.0, 1.0, 0.4, 0.05)
                bert_w = st.slider("BERT weight", 0.0, 1.0, 0.6, 0.05)

        if "Hybrid Serial" in model:
            st.markdown("---")
            st.subheader("🔀 Serial Settings")
            bm25_candidates = st.slider("BM25 candidates", 20, 200, 100, 10)

        top_k = st.slider("📄 Number of results", 5, 50, 10)

        # ── Query Refinement Options ──────────────────────────────────
        st.markdown("---")
        st.subheader("🧠 Query Formulation Assistance")

        use_spelling = st.checkbox("✏️  Spelling Correction", value=True,
                                   help="Medical-aware: corrects misspellings but protects gene names (KRAS, BRAF, etc.)")
        use_prf = st.checkbox("🔥 PRF Expansion", value=True,
                               help="Adds up to 3 corpus-derived terms from the top BM25 results")

        if use_prf:
            num_prf_terms = st.slider("Number of PRF terms", 1, 5, 3)
        else:
            num_prf_terms = 3

        st.caption("💡 Query suggestions appear below search results")

        # ── Display Options ────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🎨 Display & Clustering")
        show_text      = st.checkbox("📖 Show document text", value=True)
        max_text_len   = st.slider("📏 Max text length", 100, 1000, 300)
        show_matching  = st.checkbox("🔍 Show matching details", value=True)
        n_clusters     = st.slider("🧩 Number of clusters", 2, 8, 4, help="Number of clusters to group search results into.")

        st.markdown("---")
        _render_stats(model)
        
        st.markdown("---")
        if st.button("🔄 Clear App Cache", use_container_width=True):
            st.cache_resource.clear()
            st.success("Cache cleared! Reloading...")
            time.sleep(0.5)
            st.rerun()

    # ── Query area ────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["🔍 Search Engine", "📊 System Evaluation"])

    with tab1:
        st.header("📝 Enter your search query")

        examples = [
            "Liposarcoma CDK4 Amplification",
            "Colon cancer KRAS BRAF",
            "Meningioma NF2 AKT1",
            "Melanoma BRAF CDKN2A"
        ]
        cols = st.columns(len(examples))
        for i, ex in enumerate(examples):
            with cols[i]:
                if st.button(f"🔍 {ex}", key=f"ex_{i}"):
                    st.session_state.query = ex

        query = st.text_input("Search query:", value=st.session_state.get("query", ""),
                              placeholder="e.g., Colon cancer KRAS (G13D)...",
                              label_visibility="collapsed")

        c1, c2, c3 = st.columns([1, 1, 4])
        with c1:
            search_btn = st.button("🔍 Search", type="primary", use_container_width=True)

        # ── Search ────────────────────────────────────────────────────────────────
        if "last_search" not in st.session_state:
            st.session_state.last_search = None

        if search_btn and query:
            t0 = time.time()
            results = []
            method_used = model
            refined_query = query
            refinement_info = {}

            with st.spinner("🔍 Searching..."):
                try:
                    refiner = get_refiner()

                    prf_docs = []
                    if use_prf:
                        bm25_temp = get_bm25_service()
                        prf_docs = bm25_temp.search(query, top_k=5)

                    refined = refiner.refine(
                        query=query,
                        apply_spelling=use_spelling,
                        apply_prf=use_prf,
                        top_docs=prf_docs,
                        num_prf_terms=num_prf_terms,
                    )

                    refined_query = refined['expanded_query']
                    refinement_info = refined

                    # Default values
                    bm25_candidates = bm25_candidates if 'bm25_candidates' in dir() else 100
                    fusion = fusion if 'fusion' in dir() else "rrf"
                    bm25_w = bm25_w if 'bm25_w' in dir() else 0.4
                    bert_w = bert_w if 'bert_w' in dir() else 0.6

                    # Execute search
                    if "BM25" in model and "Hybrid" not in model:
                        svc = get_bm25_service()
                        svc.k1 = k1
                        svc.b = b
                        results = svc.search(refined_query, top_k=top_k)
                        method_used = f"BM25 (k1={k1}, b={b}) + Refinement"

                    elif "Hybrid Serial" in model:
                        svc = get_hybrid_service()
                        svc.k1 = k1
                        svc.b = b
                        results = svc.search(refined_query, mode="serial", top_k=top_k,
                                             bm25_candidates=bm25_candidates, bert_query=query)
                        method_used = f"Hybrid Serial + Refinement"

                    elif "Hybrid Parallel" in model:
                        svc = get_hybrid_service()
                        svc.k1 = k1
                        svc.b = b
                        results = svc.search(refined_query, mode="parallel", fusion=fusion,
                                             top_k=top_k, bm25_weight=bm25_w, bert_weight=bert_w,
                                             bert_query=query)
                        method_used = f"Hybrid Parallel {fusion.upper()} + Refinement"

                    elif "VSM" in model:
                        results = get_vsm_service().search(refined_query, top_k=top_k)
                        method_used = "VSM TF-IDF + Refinement"

                    elif "BERT" in model:
                        # BERT uses the original query — expanded queries dilute the dense embedding
                        results = get_bert_service().search(query, top_k=top_k)
                        method_used = "BERT Semantic"

                    else:
                        results = get_simple_service().search(refined_query, top_k=top_k)
                        method_used = "Simple TF-IDF + Refinement"

                    if results:
                        for r in results:
                            r['refinement_info'] = refinement_info

                except Exception as e:
                    st.error(f"Search error: {e}")
                    import traceback
                    st.error(traceback.format_exc())

            elapsed = time.time() - t0
            
            st.session_state.last_search = {
                "results": results,
                "method_used": method_used,
                "refined_query": refined_query,
                "refinement_info": refinement_info,
                "elapsed": elapsed,
                "query": query,
                "prf_docs": prf_docs,
            }

        # Check if we have results in session state to display
        if st.session_state.last_search is not None:
            results = st.session_state.last_search["results"]
            method_used = st.session_state.last_search["method_used"]
            refined_query = st.session_state.last_search["refined_query"]
            refinement_info = st.session_state.last_search["refinement_info"]
            elapsed = st.session_state.last_search["elapsed"]
            last_query = st.session_state.last_search["query"]
            last_prf_docs = st.session_state.last_search.get("prf_docs", [])

            if results:
                st.success(f"✅ {len(results)} results in {elapsed:.3f}s")
                st.caption(f"📐 Method: {method_used}")

                if use_spelling or use_prf:
                    with st.expander("🧠 Query Formulation Summary", expanded=False):
                        st.write(f"**Original:** `{last_query}`")
                        if refined_query != last_query:
                            st.write(f"**Refined:** `{refined_query}`")
                        if refinement_info.get('corrections_made'):
                            st.write(f"**Spelling fixes:** `{refinement_info['corrections_made']}`")
                        if refinement_info.get('prf_terms_added'):
                            st.write(f"**PRF Terms Added:** `{refinement_info['prf_terms_added']}`")

                # ── Query Suggestions ─────────────────────────────────────
                try:
                    refiner = get_refiner()
                    suggestions = refiner.suggest_queries(last_query, top_docs=last_prf_docs or results, n=3)
                    if suggestions:
                        st.markdown("**💡 Try also:**")
                        sug_cols = st.columns(len(suggestions))
                        for i, sug in enumerate(suggestions):
                            with sug_cols[i]:
                                label = sug if len(sug) <= 45 else sug[:42] + "..."
                                if st.button(f"🔍 {label}", key=f"sug_{i}", use_container_width=True):
                                    st.session_state.query = sug
                                    st.rerun()
                except Exception:
                    pass

                st.markdown("---")

                # Show tabs for Standard List vs Clustered Analysis
                view_tab1, view_tab2 = st.tabs(["📋 Standard List", "🧩 Clustered Analysis"])
                
                with view_tab1:
                    for i, r in enumerate(results, 1):
                        score = r["score"]
                        dot = "🟢" if score > 0.7 else ("🟡" if score > 0.3 else "🟠")

                        st.markdown(f"### {i}. 📄 `{r['doc_id']}`")
                        st.markdown(f"**🎯 Score:** `{dot} {score:.6f}`")

                        if "bm25_rank" in r:
                            st.caption(f"BM25 rank: {r['bm25_rank']} | BERT rank: {r['bert_rank']}")

                        if show_text and r.get("text"):
                            st.markdown("**📖 Content:**")
                            preview = r["text"][:max_text_len]
                            if len(r["text"]) > max_text_len:
                                preview += "..."
                            st.markdown(f"> {preview}")
                            with st.expander("📚 Full document"):
                                st.write(r.get("full_text", r["text"]))

                        if show_matching:
                            with st.expander("🔍 Matching details"):
                                _show_matching(r, last_query, model)

                        st.markdown("---")
                
                with view_tab2:
                    st.subheader("🧩 Clustered View (PCA Projected)")
                    # Run clustering service
                    try:
                        clustering_svc = get_clustering_service()
                        cluster_results = clustering_svc.cluster_search_results(
                            results=results,
                            n_clusters=n_clusters
                        )
                        
                        scatter_data = cluster_results["scatter_data"]
                        cluster_labels = cluster_results["cluster_labels"]
                        grouped_results = cluster_results["grouped_results"]
                        
                        if scatter_data:
                            import pandas as pd
                            import plotly.express as px
                            
                            df = pd.DataFrame(scatter_data)
                            fig = px.scatter(
                                df,
                                x="x",
                                y="y",
                                color="cluster_label",
                                hover_data={"doc_id": True, "score": ":.4f", "snippet": True, "x": False, "y": False},
                                title="2D Document Clusters (PCA)"
                            )
                            fig.update_layout(
                                legend_title_text='Clusters',
                                xaxis_title="PCA Dimension 1",
                                yaxis_title="PCA Dimension 2"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Grouped Results Expander
                            st.markdown("### 📁 Documents by Topic Group")
                            for cluster_id in sorted(cluster_labels.keys()):
                                label = cluster_labels[cluster_id]
                                docs_in_cluster = grouped_results.get(cluster_id, [])
                                
                                with st.expander(f"📂 {label} ({len(docs_in_cluster)} documents)", expanded=True):
                                    for doc in docs_in_cluster:
                                        col1, col2 = st.columns([4, 1])
                                        with col1:
                                            st.markdown(f"**📄 `{doc['doc_id']}`**")
                                        with col2:
                                            st.markdown(f"Score: `{doc['score']:.4f}`")
                                            
                                        if show_text and doc.get("text"):
                                            preview = doc["text"][:max_text_len]
                                            if len(doc["text"]) > max_text_len:
                                                preview += "..."
                                            st.markdown(f"> {preview}")
                                        st.markdown(" ")
                        else:
                            st.info("Not enough results to cluster.")
                    except Exception as e:
                        st.error(f"Error during clustering: {e}")
                        import traceback
                        st.error(traceback.format_exc())
            else:
                st.warning("⚠️ No results found.")


        # ── About ─────────────────────────────────────────────────────────────────
        with st.expander("ℹ️ About this system"):
            st.markdown("""
            ### IR System 2026 – Medical Clinical Trials Search

            **Query Formulation Assistance**
            - **Spelling Correction**: Medical-aware — protects gene names (KRAS, BRAF, EGFR, etc.)
              and mutation codes (G12D, V600E) from being "corrected" into common English words.
            - **PRF Expansion**: Corpus-aware pseudo-relevance feedback. Extracts up to 3 high-IDF
              terms co-occurring across the top BM25 results. Excludes clinical trial boilerplate.

            **Query Suggestions** (shown after every search)
            - Suggests alternative phrasings using PRF terms and medical synonyms.

            **BERT & Hybrid models**
            - BERT always encodes the *original* query (not the PRF-expanded one) so the dense
              embedding isn't diluted by loosely related PRF terms.
            - Hybrid modes use the expanded query for BM25 recall and the original for BERT.
            """)

    with tab2:
        _show_evaluation_tab()


# ── Sidebar stats ─────────────────────────────────────────────────────────────

def _render_stats(model: str):
    try:
        if "Hybrid" in model:
            svc = get_hybrid_service()
            stats = svc.get_stats()
            st.info(
                f"📊 **Hybrid Stats**\n\n"
                f"📄 Docs (BM25): {stats['bm25_documents']:,}\n"
                f"🔤 Terms: {stats['bm25_unique_terms']:,}\n"
                f"📐 Avg len: {stats['bm25_avg_length']:.1f}\n"
                f"🤖 BERT: {stats['bert_model']}\n"
                f"📐 Vector dim: {stats['bert_vector_dim']}"
            )
        elif "BM25" in model:
            svc = get_bm25_service()
            stats = svc.get_stats()
            st.info(
                f"📊 **BM25 Stats**\n\n"
                f"🔤 Unique terms: {stats['unique_terms']:,}\n"
                f"📄 Documents: {stats['total_documents']:,}\n"
                f"📐 Avg length: {stats['avg_doc_length']:.1f}"
            )
        elif "VSM" in model:
            svc = get_vsm_service()
            st.info(
                f"📊 **VSM Stats**\n\n"
                f"🔤 Terms: {len(svc.inverted_index):,}\n"
                f"📄 Docs: {svc.total_docs:,}"
            )
        elif "BERT" in model:
            svc = get_bert_service()
            stats = svc.get_stats()
            st.info(
                f"📊 **BERT Stats**\n\n"
                f"🤖 {stats['model_name']}\n"
                f"📐 dim: {stats['vector_dim']}\n"
                f"📄 Docs: {stats['total_documents']:,}"
            )
        else:
            svc = get_simple_service()
            st.info(
                f"📊 **Simple TF-IDF**\n\n"
                f"🔤 Terms: {len(svc.index):,}\n"
                f"📄 Docs: {len(svc.documents):,}"
            )
    except Exception as e:
        st.error(f"Stats error: {e}")


# ── Matching details ──────────────────────────────────────────────────────────

def _show_matching(result: dict, query: str, model: str):
    try:
        ref_info = result.get('refinement_info', {})
        if ref_info.get('expanded_query'):
            st.write(f"**Query (refined):** `{ref_info['expanded_query']}`")

        if "BERT" in model and "Hybrid" not in model:
            svc = get_bert_service()
            tokens = svc.preprocessor.process(query)
            st.write(f"**Tokens:** `{tokens}`")
            st.write(f"**Score:** `{result['score']:.6f}` (cosine similarity)")

        elif "Hybrid Serial" in model:
            svc = get_hybrid_service()
            tokens = svc.bm25.query_processor.process_query(query)
            st.write(f"**Query tokens:** `{tokens}`")
            st.write("**Stage 1:** BM25 → **Stage 2:** BERT rerank")
            st.write(f"**Final score:** `{result['score']:.6f}`")

        elif "Hybrid Parallel" in model:
            svc = get_hybrid_service()
            tokens = svc.bm25.query_processor.process_query(query)
            st.write(f"**Query tokens:** `{tokens}`")
            st.write(f"**BM25 rank:** `{result.get('bm25_rank', '-')}`")
            st.write(f"**BERT rank:** `{result.get('bert_rank', '-')}`")
            st.write(f"**Fused score:** `{result['score']:.6f}`")

        elif "VSM" in model:
            svc = get_vsm_service()
            tokens = svc.query_processor.process_query(query)
            st.write(f"**Query tokens:** `{tokens}`")
            for token in tokens:
                d = svc.get_term_details(result["doc_id"], token)
                if d["tf"] > 0 or d["idf"] > 0:
                    st.write(f"**`{token}`:** TF={d['tf']:.4f} | IDF={d['idf']:.4f} | TF-IDF={d['tfidf']:.4f}")
                else:
                    st.write(f"**`{token}`:** not found in document")

        elif "BM25" in model:
            svc = get_bm25_service()
            tokens = svc.query_processor.process_query(query)
            st.write(f"**Query tokens:** `{tokens}`")
            st.write(f"**k1={svc.k1}, b={svc.b}**")
            avg = svc.document_store.avg_doc_length
            dlen = svc.document_store.get_length(result["doc_id"])
            st.write(f"Doc length: `{dlen}` | Avg: `{avg:.1f}`")
            for token in tokens:
                tf = svc.inverted_index.get_term_frequency(token, result["doc_id"])
                df = svc.inverted_index.doc_frequency.get(token, 0)
                idf = svc.scorer.compute_idf(df, svc.document_store.total_docs)
                sc = svc.scorer.score_term(tf, dlen, avg, idf)
                if sc > 0:
                    st.write(f"**`{token}`:** TF={tf} | IDF={idf:.4f} | BM25={sc:.4f}")
                else:
                    st.write(f"**`{token}`:** not found")

        else:
            svc = get_simple_service()
            tokens = svc.query_processor.process_query(query)
            st.write(f"**Query tokens:** `{tokens}`")
            for token in tokens:
                freq = 0
                if token in svc.index:
                    td = svc.index[token]
                    if isinstance(td, dict):
                        freq = td.get(result["doc_id"], 0)
                    elif isinstance(td, list):
                        for d, f in td:
                            if str(d) == result["doc_id"]:
                                freq = f
                                break
                st.write(f"- `{token}`: {freq} occurrence(s)")

    except Exception as e:
        st.write(f"Error: {e}")


def _show_evaluation_tab():
    st.header("📊 ClinicalTrials IR System Evaluation")
    st.markdown("""
    Evaluate all retrieval models using the official relevance judgments (qrels) from **ClinicalTrials (TREC PM 2017)**.
    The evaluation compares the models **Before Improvements (Baseline)** versus **After Improvements (+ Query Refinement)**.
    """)
    
    results_path = "data/evaluation/results_clinical.json"
    
    if not os.path.exists(results_path):
        st.warning("⚠️ No evaluation results found. Please run the evaluation pipeline first.")
        if st.button("🚀 Run End-to-End Evaluation Pipeline"):
            with st.spinner("Running evaluation (this may take a few minutes)..."):
                import subprocess
                try:
                    res = subprocess.run([sys.executable, "scripts/evaluate.py"], capture_output=True, text=True, check=True)
                    st.success("✅ Evaluation pipeline completed successfully!")
                    st.cache_resource.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error running evaluation: {e}")
        return
        
    # Load results
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    st.markdown("### 📈 System Configuration & Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Test Set / Dataset", "TREC PM 2017")
    with col2:
        st.metric("Evaluable Queries", f"{data.get('eval_queries', 0)}")
    with col3:
        st.metric("Indexed Documents", f"{data.get('indexed_docs', 0):,}")
        
    # Parse models into a dataframe
    models_data = data.get("models", {})
    
    import pandas as pd
    rows = []
    for model_name, metrics in models_data.items():
        rows.append({
            "Model": model_name,
            "MAP": round(metrics.get("map", 0), 4),
            "nDCG@10": round(metrics.get("ndcg_cut_10", 0), 4),
            "P@10": round(metrics.get("P_10", 0), 4),
            "Recall@100": round(metrics.get("recall", 0), 4)
        })
    df = pd.DataFrame(rows)
    
    st.markdown("### 📊 Metric Comparison Table")
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Let's separate Baseline vs Refinement for side-by-side comparison charts
    chart_rows = []
    for model_name, metrics in models_data.items():
        if "Baseline" in model_name:
            base_name = model_name.replace(" (Baseline)", "")
            version = "Baseline (Before)"
        elif "+ Refinement" in model_name:
            base_name = model_name.replace(" (+ Refinement)", "")
            version = "Enhanced (After)"
        else:
            base_name = model_name
            version = "Standard"
            
        chart_rows.append({
            "Model": base_name,
            "Version": version,
            "MAP": metrics.get("map", 0),
            "nDCG@10": metrics.get("ndcg_cut_10", 0),
            "P@10": metrics.get("P_10", 0),
            "Recall@100": metrics.get("recall", 0)
        })
        
    df_chart = pd.DataFrame(chart_rows)
    
    st.markdown("### 📈 Visual Performance Comparison")
    
    metric_to_plot = st.selectbox("Select metric to visualize", ["MAP", "nDCG@10", "P@10", "Recall@100"])
    
    # Pivot for side-by-side plotting: index=Model, columns=Version, values=metric
    df_pivot = df_chart.pivot(index="Model", columns="Version", values=metric_to_plot)
    
    st.bar_chart(df_pivot, use_container_width=True)
    
    # Highlight achievements
    st.markdown("### 💡 Key Evaluation Findings")
    best_base_model = ""
    best_base_score = -1
    best_enh_model = ""
    best_enh_score = -1
    
    for row in chart_rows:
        score = row[metric_to_plot]
        if row["Version"] == "Baseline (Before)":
            if score > best_base_score:
                best_base_score = score
                best_base_model = row["Model"]
        elif row["Version"] == "Enhanced (After)":
            if score > best_enh_score:
                best_enh_score = score
                best_enh_model = row["Model"]
                
    st.info(f"🏆 **Best Performing Model ({metric_to_plot}):**")
    st.markdown(f"- **Baseline (Before Improvements):** `{best_base_model}` with `{best_base_score:.4f}`")
    st.markdown(f"- **Enhanced (After Improvements):** `{best_enh_model}` with `{best_enh_score:.4f}`")
    
    st.markdown("---")
    if st.button("🔄 Re-run Evaluation Pipeline", key="btn_rerun_eval"):
        with st.spinner("Re-running evaluation (this may take a few minutes)..."):
            import subprocess
            try:
                subprocess.run([sys.executable, "scripts/evaluate.py"], capture_output=True, text=True, check=True)
                st.success("✅ Evaluation completed and updated!")
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")


if __name__ == "__main__":
    main()