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
    model_key = os.getenv("BERT_MODEL_KEY", "fast")
    return BERTSearchService(model_key=model_key)


@st.cache_resource
def get_hybrid_service():
    return HybridSearchService(
        bm25_service=get_bm25_service(),
        bert_service=get_bert_service(),
    )


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

            with st.spinner("🔍 Searching..."):
                try:
                    # Default values
                    bm25_candidates = bm25_candidates if 'bm25_candidates' in locals() else 100
                    fusion = fusion if 'fusion' in locals() else "rrf"
                    bm25_w = bm25_w if 'bm25_w' in locals() else 0.4
                    bert_w = bert_w if 'bert_w' in locals() else 0.6

                    # Execute search with original query (NO Refinement)
                    if "BM25" in model and "Hybrid" not in model:
                        svc = get_bm25_service()
                        svc.k1 = k1
                        svc.b = b
                        results = svc.search(query, top_k=top_k)
                        method_used = f"BM25 (k1={k1}, b={b})"

                    elif "Hybrid Serial" in model:
                        svc = get_hybrid_service()
                        svc.k1 = k1
                        svc.b = b
                        results = svc.search(query, mode="serial", top_k=top_k,
                                             bm25_candidates=bm25_candidates)
                        method_used = f"Hybrid Serial"

                    elif "Hybrid Parallel" in model:
                        svc = get_hybrid_service()
                        svc.k1 = k1
                        svc.b = b
                        results = svc.search(query, mode="parallel", fusion=fusion,
                                             top_k=top_k, bm25_weight=bm25_w, bert_weight=bert_w)
                        method_used = f"Hybrid Parallel {fusion.upper()}"

                    elif "VSM" in model:
                        results = get_vsm_service().search(query, top_k=top_k)
                        method_used = "VSM TF-IDF"

                    elif "BERT" in model:
                        results = get_bert_service().search(query, top_k=top_k)
                        method_used = "BERT Semantic"

                    else:
                        results = get_simple_service().search(query, top_k=top_k)
                        method_used = "Simple TF-IDF"

                except Exception as e:
                    st.error(f"Search error: {e}")
                    import traceback
                    st.error(traceback.format_exc())

            elapsed = time.time() - t0
            
            st.session_state.last_search = {
                "results": results,
                "method_used": method_used,
                "elapsed": elapsed,
                "query": query,
            }

        # Check if we have results in session state to display
        if st.session_state.last_search is not None:
            results = st.session_state.last_search["results"]
            method_used = st.session_state.last_search["method_used"]
            elapsed = st.session_state.last_search["elapsed"]
            last_query = st.session_state.last_search["query"]

            if results:
                st.success(f"✅ {len(results)} results in {elapsed:.3f}s")
                st.caption(f"📐 Method: {method_used}")

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
                            import numpy as np
                            import plotly.graph_objects as go
                            
                            df = pd.DataFrame(scatter_data)
                            
                            # Map doc_id to a simple short ID (D1, D2, D3...)
                            doc_id_to_short = {doc["doc_id"]: f"D{idx}" for idx, doc in enumerate(results, 1)}
                            df["short_id"] = df["doc_id"].map(doc_id_to_short)
                            
                            # Define beautiful curated colors
                            CLUSTER_STYLES = {
                                0: {"color": "#2563EB", "bg_color": "#EFF6FF", "border_color": "#BFDBFE", "text_color": "#1E40AF"},
                                1: {"color": "#059669", "bg_color": "#ECFDF5", "border_color": "#A7F3D0", "text_color": "#065F46"},
                                2: {"color": "#DC2626", "bg_color": "#FEF2F2", "border_color": "#FCA5A5", "text_color": "#991B1B"},
                                3: {"color": "#D97706", "bg_color": "#FFFBEB", "border_color": "#FDE68A", "text_color": "#92400E"},
                                4: {"color": "#7C3AED", "bg_color": "#F5F3FF", "border_color": "#DDD6FE", "text_color": "#5B21B6"},
                                5: {"color": "#DB2777", "bg_color": "#FDF2F8", "border_color": "#FBCFE8", "text_color": "#9D174D"},
                                6: {"color": "#0891B2", "bg_color": "#ECFEFF", "border_color": "#CFFAFE", "text_color": "#155E75"},
                                7: {"color": "#0D9488", "bg_color": "#F0FDF4", "border_color": "#CCFBF1", "text_color": "#115E59"}
                            }
                            
                            def get_cluster_emoji(label_str: str) -> str:
                                label_lower = label_str.lower()
                                if any(w in label_lower for w in ["cancer", "tumor", "carcinoma", "melanoma", "lymphoma", "leukemia", "sarcoma", "oncology", "breast", "lung", "colon"]):
                                    return "🎗️"
                                elif any(w in label_lower for w in ["gene", "dna", "rna", "kras", "braf", "egfr", "her2", "mutat"]):
                                    return "🧬"
                                elif any(w in label_lower for w in ["drug", "therap", "chemo", "immunother", "regimen"]):
                                    return "💊"
                                elif any(w in label_lower for w in ["pediatric", "child", "children", "boy", "girl"]):
                                    return "🧒"
                                elif any(w in label_lower for w in ["cardiac", "heart", "coronary", "arter"]):
                                    return "❤️"
                                elif any(w in label_lower for w in ["brain", "neuro", "meningi"]):
                                    return "🧠"
                                elif any(w in label_lower for w in ["food", "diet", "nutrition", "eat", "meal"]):
                                    return "🥗"
                                elif any(w in label_lower for w in ["sport", "exercise", "train", "physic"]):
                                    return "⚽"
                                else:
                                    return "📋"
                            
                            # Compute overall range for adaptive ellipse sizing
                            x_range = max(df["x"].max() - df["x"].min(), 0.1)
                            y_range = max(df["y"].max() - df["y"].min(), 0.1)
                            default_a = max(x_range * 0.08, 0.15)
                            default_b = max(y_range * 0.08, 0.15)
                            
                            # Create Plotly figure
                            fig = go.Figure()
                            
                            # 1. Add cluster points, ellipses, and centers
                            cluster_ids = sorted(df["cluster_id"].unique())
                            for cid in cluster_ids:
                                cluster_df = df[df["cluster_id"] == cid]
                                style = CLUSTER_STYLES.get(cid % len(CLUSTER_STYLES))
                                color = style["color"]
                                
                                label_str = cluster_labels[cid]
                                keywords = label_str.split(":", 1)[1].strip() if ":" in label_str else label_str
                                clean_label = f"Cluster {cid + 1} ({keywords.title()})"
                                
                                fig.add_trace(go.Scatter(
                                    x=cluster_df["x"],
                                    y=cluster_df["y"],
                                    mode="markers+text",
                                    marker=dict(
                                        size=11,
                                        color=color,
                                        line=dict(width=1, color="white")
                                    ),
                                    text=cluster_df["short_id"],
                                    textposition="top center",
                                    textfont=dict(size=10, family="Outfit, sans-serif", color="#374151"),
                                    name=clean_label,
                                    hovertext=cluster_df.apply(lambda r: f"<b>{r['short_id']} ({r['doc_id']})</b><br>Score: {r['score']:.4f}<br>{r['snippet']}", axis=1),
                                    hoverinfo="text",
                                    showlegend=True
                                ))
                                
                                # Calculate center of this cluster
                                xc = cluster_df["x"].mean()
                                yc = cluster_df["y"].mean()
                                
                                # Add cluster center 'X' marker
                                fig.add_trace(go.Scatter(
                                    x=[xc],
                                    y=[yc],
                                    mode="markers",
                                    marker=dict(
                                        symbol="x",
                                        size=14,
                                        color=color,
                                        line=dict(width=2)
                                    ),
                                    showlegend=False,
                                    hoverinfo="skip"
                                ))
                                
                                # Calculate and add cluster boundary ellipse
                                if len(cluster_df) >= 2:
                                    xmin, xmax = cluster_df["x"].min(), cluster_df["x"].max()
                                    ymin, ymax = cluster_df["y"].min(), cluster_df["y"].max()
                                    a = max((xmax - xmin) / 2 * 1.3, default_a)
                                    b = max((ymax - ymin) / 2 * 1.3, default_b)
                                else:
                                    a, b = default_a, default_b
                                    
                                fig.add_shape(
                                    type="circle",
                                    xref="x", yref="y",
                                    x0=xc - a, y0=yc - b,
                                    x1=xc + a, y1=yc + b,
                                    line=dict(color=color, width=1, dash="dot"),
                                    fillcolor=color,
                                    opacity=0.08,
                                )
                                
                            # 2. Add Cluster Center dummy trace for the legend
                            fig.add_trace(go.Scatter(
                                x=[None],
                                y=[None],
                                mode="markers",
                                marker=dict(symbol="x", size=10, color="#374151", line=dict(width=2)),
                                name="Cluster Center",
                                showlegend=True
                            ))
                            
                            # 3. Update layout
                            fig.update_layout(
                                title=dict(
                                    text="Document Clustering using K-Means",
                                    font=dict(size=18, family="Outfit, sans-serif", color="#111827"),
                                    x=0.5,
                                    xanchor="center"
                                ),
                                xaxis=dict(
                                    title="PCA Component 1",
                                    gridcolor="rgba(0,0,0,0.05)",
                                    zerolinecolor="rgba(0,0,0,0.1)",
                                    showgrid=True,
                                    zeroline=True,
                                    titlefont=dict(family="Outfit, sans-serif", size=11)
                                ),
                                yaxis=dict(
                                    title="PCA Component 2",
                                    gridcolor="rgba(0,0,0,0.05)",
                                    zerolinecolor="rgba(0,0,0,0.1)",
                                    showgrid=True,
                                    zeroline=True,
                                    titlefont=dict(family="Outfit, sans-serif", size=11)
                                ),
                                plot_bgcolor="rgba(249, 250, 251, 0.6)",
                                paper_bgcolor="rgba(0,0,0,0)",
                                legend=dict(
                                    bgcolor="rgba(255,255,255,0.9)",
                                    bordercolor="rgba(229, 231, 235, 1)",
                                    borderwidth=1,
                                    font=dict(size=10, family="Outfit, sans-serif")
                                ),
                                margin=dict(l=40, r=40, t=50, b=40),
                                hoverlabel=dict(
                                    font_size=12,
                                    font_family="Outfit, sans-serif"
                                )
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # 4. Render Grid of Cluster Cards
                            st.markdown("<h3 style='font-family: Outfit, sans-serif; text-align: center; margin-bottom: 20px;'>📁 Cluster Content Overview</h3>", unsafe_allow_html=True)
                            
                            max_cols_per_row = 3
                            for i in range(0, len(cluster_ids), max_cols_per_row):
                                row_cids = cluster_ids[i : i + max_cols_per_row]
                                cols = st.columns(len(row_cids))
                                for idx, cid in enumerate(row_cids):
                                    with cols[idx]:
                                        style = CLUSTER_STYLES.get(cid % len(CLUSTER_STYLES))
                                        label_str = cluster_labels[cid]
                                        keywords = label_str.split(":", 1)[1].strip() if ":" in label_str else label_str
                                        cluster_title = f"Cluster {cid + 1} ({keywords.title()})"
                                        cluster_emoji = get_cluster_emoji(keywords)
                                        
                                        docs_in_cluster = grouped_results.get(cid, [])
                                        
                                        doc_list_html = ""
                                        for doc in docs_in_cluster:
                                            short_id = doc_id_to_short[doc["doc_id"]]
                                            snippet = doc.get("text", "")
                                            
                                            if len(snippet) > 85:
                                                snippet = snippet[:82] + "..."
                                                
                                            doc_list_html += f"""
                                            <div style="display: flex; margin-bottom: 12px; font-family: 'Outfit', sans-serif; font-size: 0.88rem; line-height: 1.4; color: #374151; align-items: flex-start;">
                                                <span style="color: {style['text_color']}; font-weight: 700; min-width: 28px; display: inline-block; margin-right: 6px;">{short_id}</span>
                                                <span style="flex-grow: 1;">{snippet} <code style="font-size: 0.72rem; color: #9CA3AF;">({doc['doc_id']})</code></span>
                                            </div>
                                            """
                                            
                                        if not doc_list_html:
                                            doc_list_html = "<div style='color: #9CA3AF; font-style: italic; font-size: 0.85rem;'>No documents in this cluster</div>"
                                            
                                        card_html = f"""
                                        <div style="border: 1px solid {style['border_color']}; border-radius: 12px; background-color: white; margin-bottom: 20px; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); min-height: 220px; display: flex; flex-direction: column;">
                                            <div style="background-color: {style['bg_color']}; padding: 12px 16px; border-bottom: 1px solid {style['border_color']}; display: flex; justify-content: space-between; align-items: center;">
                                                <span style="color: {style['text_color']}; font-weight: 700; font-size: 0.92rem; font-family: 'Outfit', sans-serif; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 80%;">{cluster_title}</span>
                                                <span style="font-size: 1.2rem;">{cluster_emoji}</span>
                                            </div>
                                            <div style="padding: 16px; flex-grow: 1; display: flex; flex-direction: column; justify-content: flex-start;">
                                                {doc_list_html}
                                            </div>
                                        </div>
                                        """
                                        st.markdown(card_html, unsafe_allow_html=True)
                                        
                            st.markdown("""
                                <div style="text-align: center; margin-top: 15px; margin-bottom: 25px;">
                                    <span style="background-color: #F3F4F6; border: 1px dashed #D1D5DB; border-radius: 8px; padding: 8px 18px; font-size: 0.85rem; color: #4B5563; font-weight: 500; font-family: 'Outfit', sans-serif; display: inline-block; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                                        📝 Documents in the same cluster are more similar to each other.
                                    </span>
                                </div>
                            """, unsafe_allow_html=True)
                            
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

            **Retrieval Models**
            - **BM25**: Best Matching 25 with configurable k1 and b parameters
            - **VSM TF-IDF**: Vector Space Model with cosine similarity
            - **BERT Embeddings**: Semantic search using Sentence-BERT
            - **Hybrid Serial**: BM25 retrieves candidates → BERT reranks
            - **Hybrid Parallel**: BM25 + BERT results fused via RRF or linear combination

            **Clustering**
            - K-Means clustering on BERT embeddings
            - PCA projection for 2D visualization
            - Interactive Plotly chart with cluster boundaries

            **Dataset**
            - ClinicalTrials.gov (TREC PM 2017)
            - 241,006 indexed documents
            - 30 evaluation queries with relevance judgments
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
    The evaluation shows performance metrics for each model on the test queries.
    """)
    
    # Try to load baseline-only results first, then fallback to full results
    results_path = "data/evaluation/results_clinical_baseline_only.json"
    
    if not os.path.exists(results_path):
        # Fallback to the original results file
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
    
    # Chart for visual comparison
    chart_rows = []
    for model_name, metrics in models_data.items():
        chart_rows.append({
            "Model": model_name,
            "MAP": metrics.get("map", 0),
            "nDCG@10": metrics.get("ndcg_cut_10", 0),
            "P@10": metrics.get("P_10", 0),
            "Recall@100": metrics.get("recall", 0)
        })
        
    df_chart = pd.DataFrame(chart_rows)
    
    st.markdown("### 📈 Visual Performance Comparison")
    
    metric_to_plot = st.selectbox("Select metric to visualize", ["MAP", "nDCG@10", "P@10", "Recall@100"])
    
    # Sort by selected metric descending
    df_chart_sorted = df_chart.sort_values(by=metric_to_plot, ascending=False)
    
    st.bar_chart(df_chart_sorted.set_index("Model")[metric_to_plot], use_container_width=True)
    
    # Highlight achievements
    st.markdown("### 💡 Key Evaluation Findings")
    
    best_model = df_chart.loc[df_chart[metric_to_plot].idxmax()]
    
    st.info(f"🏆 **Best Performing Model ({metric_to_plot}):**")
    st.markdown(f"- **Model:** `{best_model['Model']}`")
    st.markdown(f"- **Score:** `{best_model[metric_to_plot]:.4f}`")
    
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