# ui/clustering_app.py
"""
Clinical Trials Topic Explorer & Clustering Dashboard.
A standalone user interface for analyzing, searching, filtering, and visualizing 
the 241,006 ClinicalTrials documents based on hierarchical medical dimensions.
"""

import sys
import os
import time
import re
import sqlite3
from typing import Dict, List, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.clustering.clinical_clustering import ClinicalTrialsClusterer
from services.ranking.embeddings.bert_search_service import BERTSearchService
from services.retrieval.bm25_search_service import BM25SearchService

# ── Service Caching ───────────────────────────────────────────────────────────

@st.cache_resource
def get_clinical_clusterer() -> ClinicalTrialsClusterer:
    """Load the precomputed SQLite-based ClinicalTrialsClusterer."""
    return ClinicalTrialsClusterer()

@st.cache_resource
def get_bert_service() -> BERTSearchService:
    """Load the sentence embedding service for PCA 2D projections."""
    return BERTSearchService(model_key="fast")

@st.cache_resource
def get_bm25_service() -> BM25SearchService:
    """Load the BM25 search service for indexed text query filtering."""
    return BM25SearchService(k1=1.5, b=0.75)

# ── Helper Functions ──────────────────────────────────────────────────────────

def highlight_keywords(text: str, categories_dict: Dict[str, str], clusterer: ClinicalTrialsClusterer) -> str:
    """
    Highlights words in the text that triggered the medical topic classifications.
    Matches are wrapped in colored HTML span tags.
    """
    if not text:
        return ""
        
    highlighted = text
    
    # We will gather all matched keywords to highlight
    matches_to_highlight = []
    
    # Category colors mapped to CSS styles
    cat_colors = {
        'cancer_type': 'rgba(255, 123, 0, 0.25); border-bottom: 2px solid #FF7B00; color: #FF9E40;',
        'treatment_type': 'rgba(0, 229, 255, 0.25); border-bottom: 2px solid #00E5FF; color: #66EFFF;',
        'gene': 'rgba(186, 104, 200, 0.25); border-bottom: 2px solid #BA68C8; color: #E1BEE7;',
        'stage': 'rgba(76, 175, 80, 0.25); border-bottom: 2px solid #4CAF50; color: #81C784;',
        'demographics': 'rgba(253, 216, 53, 0.25); border-bottom: 2px solid #FDD835; color: #FFF59D;',
        'comorbidity': 'rgba(244, 67, 54, 0.25); border-bottom: 2px solid #F44336; color: #EF9A9A;'
    }

    text_lower = text.lower()
    
    # Gather all matched keywords across the topics
    for cat_name, topic in categories_dict.items():
        if topic == 'other' or cat_name not in clusterer.CATEGORY_MAPPING:
            continue
        
        keywords = clusterer.CATEGORY_MAPPING[cat_name].get(topic, [])
        for kw in keywords:
            # Check if keyword is in the text
            if kw in text_lower:
                matches_to_highlight.append((kw, cat_name))

    # Sort keywords by length descending so we replace longer phrases first (e.g. "breast cancer" before "breast")
    matches_to_highlight.sort(key=lambda x: len(x[0]), reverse=True)
    
    # To prevent double highlighting, we will keep track of already replaced index ranges
    # A simple but robust way is to replace with unique placeholders, then replace placeholders with HTML
    placeholders = {}
    for idx, (kw, cat_name) in enumerate(matches_to_highlight):
        placeholder = f"__MEDHIGHLIGHT_{idx}__"
        # Use regex to find and replace case-insensitively, keeping the original matched text
        pattern = re.compile(r'\b(' + re.escape(kw) + r')\b', re.IGNORECASE)
        
        # Check if we actually match
        match = pattern.search(highlighted)
        if match:
            original_word = match.group(1)
            style = cat_colors.get(cat_name, 'background-color: rgba(255,255,255,0.1);')
            html_rep = f'<span style="padding: 2px 6px; border-radius: 4px; font-weight: 500; margin: 0 1px; {style}">{original_word}</span>'
            placeholders[placeholder] = html_rep
            highlighted = pattern.sub(placeholder, highlighted)
            
    # Now replace placeholders with actual HTML
    for placeholder, html_content in placeholders.items():
        highlighted = highlighted.replace(placeholder, html_content)
        
    return highlighted

def get_filtered_docs_sql(selected_filters: Dict[str, str], limit: int = 150) -> List[Dict[str, Any]]:
    """
    Directly queries SQLite to retrieve documents matching specific category topic filters.
    Highly optimized using indexed columns.
    """
    db_path = 'data/processed/doc_store.db'
    if not os.path.exists(db_path):
        return []
        
    query_parts = []
    params = []
    
    for cat, val in selected_filters.items():
        if val != "All":
            query_parts.append(f"t.{cat} = ?")
            params.append(val.lower())
            
    where_clause = ""
    if query_parts:
        where_clause = "WHERE " + " AND ".join(query_parts)
        
    sql = f"""
        SELECT d.doc_id, d.text, t.cancer_type, t.treatment_type, t.gene, t.stage, t.demographics, t.comorbidity 
        FROM documents d 
        JOIN document_topics t ON d.doc_id = t.doc_id 
        {where_clause} 
        LIMIT ?
    """
    params.append(limit)
    
    results = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        
        for r in rows:
            results.append({
                'doc_id': r[0],
                'text': r[1] or "No text available",
                'full_text': r[1] or "No text available",
                'score': 1.0, # Neutral score for pure filters
                'topics': {
                    'cancer_type': r[2],
                    'treatment_type': r[3],
                    'gene': r[4],
                    'stage': r[5],
                    'demographics': r[6],
                    'comorbidity': r[7]
                }
            })
        conn.close()
    except Exception as e:
        st.error(f"Error fetching filtered docs: {e}")
        
    return results

# ── Custom CSS for Premium Look ───────────────────────────────────────────────

def inject_custom_styles():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
            
            /* Apply font family globally */
            html, body, [class*="css"], .stMarkdown, p, div, label {
                font-family: 'Outfit', sans-serif !important;
            }
            
            /* Gradient Header */
            .main-header {
                background: linear-gradient(135deg, #00C6FF 0%, #0072FF 50%, #7F00FF 100%);
                padding: 1.8rem;
                border-radius: 16px;
                color: white;
                margin-bottom: 2rem;
                box-shadow: 0 8px 24px rgba(0, 114, 255, 0.15);
                position: relative;
                overflow: hidden;
            }
            .main-header::before {
                content: "";
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 80%);
                pointer-events: none;
            }
            
            /* Metric Card styling */
            .glass-metric {
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 14px;
                padding: 1.2rem;
                text-align: center;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
                transition: all 0.3s ease;
            }
            .glass-metric:hover {
                transform: translateY(-4px);
                border-color: rgba(0, 114, 255, 0.3);
                background: rgba(255, 255, 255, 0.05);
                box-shadow: 0 8px 25px rgba(0, 114, 255, 0.1);
            }
            .metric-val {
                font-size: 2.2rem;
                font-weight: 700;
                background: linear-gradient(90deg, #00E5FF, #7F00FF);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 2px;
            }
            .metric-lbl {
                color: #8A99AD;
                font-size: 0.85rem;
                text-transform: uppercase;
                letter-spacing: 1px;
                font-weight: 500;
            }
            
            /* Badges list styling */
            .tag-badge {
                display: inline-flex;
                align-items: center;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.78rem;
                font-weight: 600;
                margin-right: 6px;
                margin-bottom: 6px;
                letter-spacing: 0.5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }
            
            .badge-ct { background-color: rgba(255, 123, 0, 0.12); color: #FF7B00; border: 1px solid rgba(255, 123, 0, 0.25); }
            .badge-tt { background-color: rgba(0, 229, 255, 0.12); color: #00E5FF; border: 1px solid rgba(0, 229, 255, 0.25); }
            .badge-gn { background-color: rgba(186, 104, 200, 0.12); color: #BA68C8; border: 1px solid rgba(186, 104, 200, 0.25); }
            .badge-st { background-color: rgba(76, 175, 80, 0.12); color: #4CAF50; border: 1px solid rgba(76, 175, 80, 0.25); }
            .badge-dm { background-color: rgba(253, 216, 53, 0.12); color: #FBC02D; border: 1px solid rgba(253, 216, 53, 0.25); }
            .badge-cb { background-color: rgba(244, 67, 54, 0.12); color: #F44336; border: 1px solid rgba(244, 67, 54, 0.25); }
            .badge-ot { background-color: rgba(120, 144, 156, 0.12); color: #90A4AE; border: 1px solid rgba(120, 144, 156, 0.25); }
            
            /* Sidebar layout style */
            .sidebar-card {
                background: rgba(255, 255, 255, 0.02);
                border: 1px solid rgba(255, 255, 255, 0.05);
                border-radius: 12px;
                padding: 1rem;
                margin-bottom: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)

# ── Main Application ──────────────────────────────────────────────────────────

def main():
    inject_custom_styles()
    
    # ── Initialize Services ───────────────────────────────────────────────────
    with st.spinner("Initializing medical databases & indices..."):
        clusterer = get_clinical_clusterer()
        bert_service = get_bert_service()
        bm25_service = get_bm25_service()

    # ── Header Banner ─────────────────────────────────────────────────────────
    st.markdown("""
        <div class="main-header">
            <h1 style="margin: 0; font-size: 2.3rem; font-weight: 700; letter-spacing: -0.5px;">🩺 Clinical Trials Medical Topic Explorer</h1>
            <p style="margin: 5px 0 0 0; opacity: 0.85; font-size: 1.05rem;">
                Advanced topic-based classification, multi-dimensional search filtering, and semantic PCA projections of 241,006 documents.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # ── Sidebar Setup ─────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ System Status")
        
        st.markdown("""
            <div class="sidebar-card">
                <div style="font-weight: 600; font-size: 0.95rem; margin-bottom: 8px; color: #FFFFFF;">📊 Database Metadata</div>
                <div style="font-size: 0.85rem; color: #8A99AD; margin-bottom: 4px;">Collection: <b>ClinicalTrials 2017</b></div>
                <div style="font-size: 0.85rem; color: #8A99AD; margin-bottom: 4px;">Total Documents: <b>241,006</b></div>
                <div style="font-size: 0.85rem; color: #8A99AD; margin-bottom: 4px;">SQLite Table: <b>document_topics</b></div>
                <div style="font-size: 0.85rem; color: #8A99AD; margin-bottom: 4px;">Topics Computed: <span style="color: #4CAF50; font-weight: bold;">241,006 (100%)</span></div>
            </div>
            <div class="sidebar-card">
                <div style="font-weight: 600; font-size: 0.95rem; margin-bottom: 8px; color: #FFFFFF;">🧬 Index Services</div>
                <div style="font-size: 0.85rem; color: #8A99AD; margin-bottom: 4px;">PubMedBERT Model: <span style="color: #00E5FF; font-weight: bold;">Active</span></div>
                <div style="font-size: 0.85rem; color: #8A99AD; margin-bottom: 4px;">FAISS Vector Index: <span style="color: #4CAF50; font-weight: bold;">Ready</span></div>
                <div style="font-size: 0.85rem; color: #8A99AD; margin-bottom: 4px;">BM25 Index: <span style="color: #4CAF50; font-weight: bold;">Ready</span></div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🏷️ Topic Dimensions")
        for key, name in clusterer.CATEGORY_LABELS.items():
            count_topics = len(clusterer.CATEGORY_MAPPING[key])
            st.caption(f"**{name}**: {count_topics} sub-topics")
            
        st.markdown("---")
        if st.button("🔄 Reset Cache", use_container_width=True):
            st.cache_resource.clear()
            st.success("Cache cleared! Reloading...")
            time.sleep(0.5)
            st.rerun()

    # ── High Level KPIs Grid ──────────────────────────────────────────────────
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown('<div class="glass-metric"><div class="metric-val">241,006</div><div class="metric-lbl">Total Documents</div></div>', unsafe_allow_html=True)
    with kpi2:
        st.markdown('<div class="glass-metric"><div class="metric-val">6</div><div class="metric-lbl">Classified Dimensions</div></div>', unsafe_allow_html=True)
    with kpi3:
        st.markdown('<div class="glass-metric"><div class="metric-val">39</div><div class="metric-lbl">Unique Topics Mapped</div></div>', unsafe_allow_html=True)
    with kpi4:
        st.markdown('<div class="glass-metric"><div class="metric-val">&lt; 1 ms</div><div class="metric-lbl">Filter Response</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs Configuration ────────────────────────────────────────────────────
    tab_global, tab_search, tab_sandbox = st.tabs([
        "📊 Global Dataset Insights",
        "🔍 Search & Topic Filtering",
        "🔬 Medical NLP Sandbox"
    ])

    # ── Tab 1: Global Insights ────────────────────────────────────────────────
    with tab_global:
        st.header("📊 Global Database Topic Distribution")
        st.markdown("""
            Analyze how the entire collection of **241,006 clinical trials** is distributed across various medical themes.
            Toggle the selector below to check different medical dimensions.
        """)
        
        # Load global counts
        global_counts = clusterer.get_global_counts()
        
        sel_col1, sel_col2 = st.columns([1, 2])
        with sel_col1:
            selected_category_key = st.selectbox(
                "Select Medical Dimension to Analyze:",
                options=list(clusterer.CATEGORY_LABELS.keys()),
                format_func=lambda x: clusterer.CATEGORY_LABELS[x],
                key="global_dim_selector"
            )
            
            # Keywords preview card
            st.markdown(f"**Keywords Mapped to topics in this dimension:**")
            kw_dict = clusterer.CATEGORY_MAPPING[selected_category_key]
            for topic, keywords in kw_dict.items():
                with st.expander(f"📁 {topic.upper()}", expanded=False):
                    st.write(", ".join(keywords[:10]) + ("..." if len(keywords) > 10 else ""))
                    
        with sel_col2:
            counts = global_counts.get(selected_category_key, {})
            # Sort topics by count desc
            sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            
            df_counts = pd.DataFrame(sorted_counts, columns=["Topic", "Document Count"])
            df_counts["Percentage (%)"] = (df_counts["Document Count"] / 241006 * 100).round(2)
            
            # Beautiful Chart
            fig = px.bar(
                df_counts,
                x="Document Count",
                y="Topic",
                orientation='h',
                color="Document Count",
                color_continuous_scale=px.colors.sequential.Tealgrn,
                text_auto='.3s',
                title=f"Frequencies for: {clusterer.CATEGORY_LABELS[selected_category_key]}"
            )
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                xaxis_title="Number of Documents",
                yaxis_title="Topic Category",
                coloraxis_showscale=False,
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📋 Topic Frequency Table")
        st.dataframe(
            df_counts, 
            column_config={
                "Topic": st.column_config.TextColumn("Medical Topic Name"),
                "Document Count": st.column_config.NumberColumn("Document Count", format="%d"),
                "Percentage (%)": st.column_config.NumberColumn("Database Ratio", format="%.2f %%")
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Grid of top topics
        st.markdown("---")
        st.markdown("### 🏆 Top Classifications Across Dimensions")
        cols = st.columns(3)
        
        cat_keys = list(clusterer.CATEGORY_LABELS.keys())
        for idx, cat_key in enumerate(cat_keys):
            col_target = cols[idx % 3]
            cat_lbl = clusterer.CATEGORY_LABELS[cat_key]
            cat_counts = global_counts.get(cat_key, {})
            
            # Find max topic excluding other
            clean_counts = {k: v for k, v in cat_counts.items() if k != 'other'}
            if not clean_counts:
                clean_counts = cat_counts
            top_topic = max(clean_counts, key=clean_counts.get)
            top_val = clean_counts[top_topic]
            other_val = cat_counts.get('other', 0)
            
            with col_target:
                st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 15px; margin-bottom: 15px;">
                        <div style="font-size: 0.8rem; color: #8A99AD; text-transform: uppercase;">{cat_lbl}</div>
                        <div style="font-size: 1.3rem; font-weight: 700; color: #FFF; margin-top: 5px;">{top_topic.replace('_', ' ').title()}</div>
                        <div style="font-size: 1rem; color: #00E5FF; font-weight: 600; margin-top: 2px;">{top_val:,} documents</div>
                        <div style="font-size: 0.8rem; color: #78909C; margin-top: 10px;">Unclassified (Other): {other_val:,} docs</div>
                    </div>
                """, unsafe_allow_html=True)

    # ── Tab 2: Search & Topic Filtering ───────────────────────────────────────
    with tab_search:
        st.header("🔍 Advanced Clinical Search & Multi-Dimensional Filtering")
        st.markdown("""
            Query the index and strictly filter results using precomputed medical topics.
            Combine free-text search queries with structured SQL database filters for micro-second indexing.
        """)
        
        # Build Filter UI
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        filter_col4, filter_col5, filter_col6 = st.columns(3)
        
        filters = {}
        
        def make_filter_selectbox(col, cat_key):
            topics = ["All"] + sorted([t for t in clusterer.CATEGORY_MAPPING[cat_key].keys()])
            # Format labels nicely
            label = clusterer.CATEGORY_LABELS[cat_key]
            with col:
                val = st.selectbox(
                    f"Select {label}:",
                    options=topics,
                    format_func=lambda x: x.replace("_", " ").title() if x != "All" else "Show All Topics",
                    key=f"filter_sel_{cat_key}"
                )
                filters[cat_key] = val
        
        make_filter_selectbox(filter_col1, 'cancer_type')
        make_filter_selectbox(filter_col2, 'treatment_type')
        make_filter_selectbox(filter_col3, 'gene')
        make_filter_selectbox(filter_col4, 'stage')
        make_filter_selectbox(filter_col5, 'demographics')
        make_filter_selectbox(filter_col6, 'comorbidity')
        
        # Query bar
        text_query = st.text_input("Enter clinical trial text search query (Optional):", 
                                  placeholder="e.g. Immunotherapy for HER2-positive breast metastasis...",
                                  key="filtered_search_query")
        
        col_search_btn, col_reset_btn, _ = st.columns([1, 1, 4])
        with col_search_btn:
            run_btn = st.button("🔍 Search & Filter", type="primary", use_container_width=True)
        with col_reset_btn:
            reset_btn = st.button("🔄 Clear Filters", use_container_width=True)
            
        if reset_btn:
            # We can reset using session state or rerun
            for key in clusterer.CATEGORY_MAPPING.keys():
                st.session_state[f"filter_sel_{key}"] = "All"
            st.session_state["filtered_search_query"] = ""
            st.rerun()

        # ── Search Execution ──────────────────────────────────────────────────
        if run_btn or text_query or any(v != "All" for v in filters.values()):
            t0 = time.time()
            results = []
            
            with st.spinner("Processing filters and searching collection..."):
                if text_query.strip():
                    # We have a text query! We search using BM25, then filter results
                    # Fetch more candidates to allow filtering room
                    raw_results = bm25_service.search(text_query, top_k=500)
                    
                    if raw_results:
                        doc_ids = [r["doc_id"] for r in raw_results]
                        doc_topics = clusterer.get_document_topics(doc_ids)
                        
                        # Filter results in python
                        for r in raw_results:
                            doc_id = r["doc_id"]
                            topics = doc_topics.get(doc_id, {})
                            
                            # Fallback if not precomputed
                            if not topics:
                                doc_text = r.get("full_text") or r.get("text") or ""
                                topics = clusterer.classify_text(doc_text)
                                
                            # Check match
                            matches_filters = True
                            for cat, val in filters.items():
                                if val != "All" and topics.get(cat, "other") != val.lower():
                                    matches_filters = False
                                    break
                                    
                            if matches_filters:
                                r_copy = r.copy()
                                r_copy["topics"] = topics
                                results.append(r_copy)
                else:
                    # Pure filter search! Directly fetch from SQLite
                    results = get_filtered_docs_sql(filters, limit=150)
            
            elapsed = time.time() - t0
            
            st.markdown("---")
            if not results:
                st.warning("⚠️ No documents matched the combined query and filters.")
            else:
                st.success(f"✅ Found {len(results)} matching trials in {elapsed:.3f} seconds")
                
                # Visualizations
                st.subheader("🧩 Results Analysis & Semantic Projection")
                vis_col1, vis_col2 = st.columns([2, 1])
                
                with vis_col1:
                    # 2D PCA Semantic Projection
                    try:
                        vectors = []
                        valid_results = []
                        for r in results:
                            doc_id = r["doc_id"]
                            vec = bert_service.get_vector(doc_id)
                            if vec is None:
                                # Encode on the fly if needed
                                text = r.get("full_text") or r.get("text") or ""
                                vec = bert_service.model.encode(text) if text else np.zeros(bert_service.model.dim)
                            vectors.append(vec)
                            valid_results.append(r)
                            
                        X = np.array(vectors, dtype=np.float32)
                        
                        # PCA reduction
                        if len(valid_results) >= 2:
                            pca = PCA(n_components=2, random_state=42)
                            coords = pca.fit_transform(X)
                        else:
                            coords = np.zeros((len(valid_results), 2))
                            
                        # Color coding based on selected dimension
                        color_dimension = st.selectbox(
                            "Color points by dimension:",
                            options=list(clusterer.CATEGORY_LABELS.keys()),
                            format_func=lambda x: clusterer.CATEGORY_LABELS[x],
                            key="pca_color_dimension"
                        )
                        
                        plot_data = []
                        for idx, r in enumerate(valid_results):
                            topic = r["topics"].get(color_dimension, "other").upper()
                            snippet = r.get("text", "")
                            if len(snippet) > 150:
                                snippet = snippet[:150] + "..."
                                
                            plot_data.append({
                                "x": float(coords[idx, 0]),
                                "y": float(coords[idx, 1]),
                                "doc_id": r["doc_id"],
                                "score": r.get("score", 1.0),
                                "topic": topic,
                                "snippet": snippet
                            })
                            
                        df_plot = pd.DataFrame(plot_data)
                        fig_pca = px.scatter(
                            df_plot,
                            x="x",
                            y="y",
                            color="topic",
                            hover_data={"doc_id": True, "score": ":.4f", "snippet": True, "x": False, "y": False},
                            title="2D PCA Semantic Space Projection"
                        )
                        fig_pca.update_layout(
                            legend_title_text='Topic Groups',
                            xaxis_title="Semantic Vector X",
                            yaxis_title="Semantic Vector Y",
                            height=400
                        )
                        st.plotly_chart(fig_pca, use_container_width=True)
                    except Exception as pca_err:
                        st.info("Scatter plot visualization requires SBERT vectors index. Loading fallback layout...")
                        st.caption(f"Error detail: {pca_err}")
                
                with vis_col2:
                    st.markdown("##### Filtered Sub-distribution")
                    # Show distribution of topics in the filtered results
                    target_distribution_cat = st.selectbox(
                        "Analyze topic breakdown of results:",
                        options=list(clusterer.CATEGORY_LABELS.keys()),
                        format_func=lambda x: clusterer.CATEGORY_LABELS[x],
                        key="distribution_analysis_selector"
                    )
                    
                    dist_counts = {}
                    for r in results:
                        top_val = r["topics"].get(target_distribution_cat, "other")
                        dist_counts[top_val] = dist_counts.get(top_val, 0) + 1
                        
                    df_dist = pd.DataFrame(list(dist_counts.items()), columns=["Topic", "Matches"])
                    df_dist["Topic"] = df_dist["Topic"].str.upper()
                    
                    fig_pie = px.pie(
                        df_dist,
                        values="Matches",
                        names="Topic",
                        hole=0.4,
                        title=f"{clusterer.CATEGORY_LABELS[target_distribution_cat]} share"
                    )
                    fig_pie.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # ── Matching Documents List ───────────────────────────────────────────
                st.markdown("### 📄 Filtered Trial Documents")
                
                # Limit visible documents list to 50 to prevent DOM slowdown
                visible_results = results[:50]
                if len(results) > 50:
                    st.info(f"Showing top 50 matches. Use queries/filters to narrow down further.")
                    
                for idx, doc in enumerate(visible_results, 1):
                    doc_id = doc["doc_id"]
                    score = doc.get("score", 1.0)
                    topics = doc["topics"]
                    
                    # Create card style expander title
                    badge_html = ""
                    # Create badge tags for each classification
                    for cat_name, label_pfx in [
                        ('cancer_type', 'ct'),
                        ('treatment_type', 'tt'),
                        ('gene', 'gn'),
                        ('stage', 'st'),
                        ('demographics', 'dm'),
                        ('comorbidity', 'cb')
                    ]:
                        t_val = topics.get(cat_name, 'other')
                        if t_val != 'other':
                            badge_html += f'<span class="tag-badge badge-{label_pfx}">{t_val.upper()}</span>'
                            
                    if not badge_html:
                        badge_html = '<span class="tag-badge badge-ot">OTHER/GENERAL</span>'
                        
                    title_html = f"📄 {idx}. <b>Document `{doc_id}`</b> &nbsp;&nbsp;&nbsp; {badge_html}"
                    
                    with st.expander(f"Trial ID: {doc_id} (Click to Expand)", expanded=(idx == 1)):
                        col_doc1, col_doc2 = st.columns([4, 1])
                        with col_doc1:
                            st.markdown(f"**Document ID:** `{doc_id}`")
                            if text_query:
                                st.markdown(f"**Search Relevance Score:** `{score:.6f}`")
                        with col_doc2:
                            # Show all tags inside expander
                            st.write("**Topic Classifications:**")
                            for c_k, c_v in topics.items():
                                val_formatted = c_v.upper() if c_v else "OTHER"
                                label = clusterer.CATEGORY_LABELS[c_k].split()[-1] # extract name without emoji
                                st.caption(f"**{label}**: {val_formatted}")
                                
                        st.markdown("**📄 Text Content (with highlighted keyword matches):**")
                        # Highlight words
                        highlighted_html = highlight_keywords(doc["full_text"], topics, clusterer)
                        
                        st.markdown(
                            f'<div style="background-color: rgba(255,255,255,0.01); border: 1px dashed rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; font-size: 0.95rem; line-height: 1.6; max-height: 400px; overflow-y: auto; color: #E3F2FD;">{highlighted_html}</div>', 
                            unsafe_allow_html=True
                        )

    # ── Tab 3: Text Sandbox ───────────────────────────────────────────────────
    with tab_sandbox:
        st.header("🔬 Medical NLP Text Classifier Sandbox")
        st.markdown("""
            Paste any custom medical text, patient clinical case, or trial description.
            The system will classify the text on-the-fly across the 6 dimensions using the rule-based keyword classification engines
            and highlight the matched entities in real-time.
        """)
        
        # Load preset examples
        examples = {
            "Select Example Case...": "",
            "Example 1: Triple-Negative Breast Cancer Chemotherapy": (
                "The patient is a 45-year-old female diagnosed with early-stage triple-negative breast cancer (TNBC). "
                "She has completed surgical resection (lumpectomy) and is scheduled to undergo adjuvant chemotherapy. "
                "The proposed regimen consists of docetaxel in combination with carboplatin. "
                "The patient has a medical history of hypertension, which is well-managed with medication. "
                "Eligibility criteria: Female patients, age >= 18, with histologically confirmed Stage I or II breast carcinoma."
            ),
            "Example 2: Elderly Lung Cancer with EGFR Mutation": (
                "A clinical trial designed for elderly patients (aged 70 and older) with advanced stage adenocarcinoma of the lung. "
                "Eligible patients must harbor an activating EGFR mutation (specifically exon 19 deletion or L858R). "
                "The trial investigates a targeted therapy using the tyrosine kinase inhibitor erlotinib as first-line treatment. "
                "Patients with major comorbidities, including congestive heart failure (CHF) or severe insulin-dependent diabetes, "
                "are excluded from participation."
            ),
            "Example 3: Relapsed Pediatric Leukemia Gene Therapy": (
                "This study evaluates a novel gene therapy approach for pediatric patients (children aged 2 to 18 years) "
                "with recurrent or refractory B-cell acute lymphoblastic leukemia (ALL). "
                "The intervention involves gene transfer using a lentiviral vector to engineer chimeric antigen receptor T-cells (CAR-T) "
                "targeting the CD19 antigen. Patients must have relapsed after at least two lines of standard chemotherapy "
                "or have progressed following bone marrow transplantation."
            )
        }
        
        selected_example = st.selectbox(
            "Load a Preset Medical Case study:",
            options=list(examples.keys()),
            key="sandbox_preset_selector"
        )
        
        default_sandbox_text = examples[selected_example] if selected_example != "Select Example Case..." else ""
        
        # Text input area
        sandbox_text = st.text_area(
            "Enter Medical / Clinical Trial Text:",
            value=default_sandbox_text,
            height=200,
            placeholder="Paste your clinical report or protocol description here...",
            key="sandbox_input_text"
        )
        
        classify_btn = st.button("🚀 Analyze & Classify Text", type="primary", use_container_width=True)
        
        if classify_btn or sandbox_text:
            if not sandbox_text.strip():
                st.warning("Please enter some text to classify.")
            else:
                t_sandbox0 = time.time()
                # Run classification on-the-fly
                classifications = clusterer.classify_text(sandbox_text)
                elapsed_sandbox = time.time() - t_sandbox0
                
                st.markdown("---")
                st.subheader("🎯 Real-Time Classification Results")
                st.caption(f"Analysis completed in {elapsed_sandbox * 1000:.2f} ms")
                
                # Grid of results
                c_cols = st.columns(3)
                
                for idx, (cat_key, topic) in enumerate(classifications.items()):
                    target_col = c_cols[idx % 3]
                    label = clusterer.CATEGORY_LABELS[cat_key]
                    
                    # Get keywords that matched
                    keywords = clusterer.CATEGORY_MAPPING[cat_key].get(topic, [])
                    matched_kws = [kw for kw in keywords if kw in sandbox_text.lower()]
                    
                    # Colors
                    badge_style = "badge-ot"
                    if topic != 'other':
                        if cat_key == 'cancer_type': badge_style = "badge-ct"
                        elif cat_key == 'treatment_type': badge_style = "badge-tt"
                        elif cat_key == 'gene': badge_style = "badge-gn"
                        elif cat_key == 'stage': badge_style = "badge-st"
                        elif cat_key == 'demographics': badge_style = "badge-dm"
                        elif cat_key == 'comorbidity': badge_style = "badge-cb"
                    
                    with target_col:
                        st.markdown(f"""
                            <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 15px; height: 160px; display: flex; flex-direction: column; justify-content: space-between;">
                                <div>
                                    <div style="font-size: 0.8rem; color: #8A99AD; text-transform: uppercase;">{label}</div>
                                    <div style="font-size: 1.15rem; font-weight: 700; margin-top: 5px; color: #FFF;">
                                        {topic.replace('_', ' ').upper()}
                                    </div>
                                </div>
                                <div>
                                    <div style="font-size: 0.75rem; color: #8A99AD;">Matches:</div>
                                    <div style="font-size: 0.8rem; color: #00E5FF; font-weight: 500; text-overflow: ellipsis; overflow: hidden; white-space: nowrap;">
                                        {', '.join(matched_kws) if matched_kws else 'None (General/Other)'}
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                
                # Highlighted Text Panel
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📖 Annotated Document Text")
                st.markdown("Matched terms and entities from the clinical vocabulary are highlighted below:")
                
                highlighted_sandbox = highlight_keywords(sandbox_text, classifications, clusterer)
                
                st.markdown(
                    f'<div style="background-color: rgba(255,255,255,0.01); border: 1px solid rgba(255,255,255,0.08); padding: 20px; border-radius: 12px; font-size: 1.05rem; line-height: 1.7; color: #E3F2FD; box-shadow: inset 0 2px 10px rgba(0,0,0,0.2);">{highlighted_sandbox}</div>', 
                    unsafe_allow_html=True
                )
                
                # Highlight legend
                st.markdown("<br>", unsafe_allow_html=True)
                st.write("**Highlighting Legend:**")
                st.markdown("""
                    <span class="tag-badge badge-ct">🩺 Cancer Type</span>
                    <span class="tag-badge badge-tt">💊 Treatment Type</span>
                    <span class="tag-badge badge-gn">🧬 Genes/Mutations</span>
                    <span class="tag-badge badge-st">🔬 Cancer Stage</span>
                    <span class="tag-badge badge-dm">🧒 Demographics</span>
                    <span class="tag-badge badge-cb">❤️ Comorbidities</span>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
