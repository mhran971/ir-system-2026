# ui/clustering_app.py
"""
Clinical Trials Topic Explorer — Advanced Clustering with UMAP/t-SNE
Shows well-separated clusters with clear visual distinction.
"""

import sys
import os
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Try importing umap (optional, better than t-SNE)
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("UMAP not installed. Falling back to t-SNE.")
    print("To install: pip install umap-learn")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.clustering.clinical_clustering import ClinicalTrialsClusterer
from services.ranking.embeddings.bert_search_service import BERTSearchService


# ── Service Caching ───────────────────────────────────────────────────────────

@st.cache_resource
def get_clusterer():
    return ClinicalTrialsClusterer()

@st.cache_resource
def get_bert():
    return BERTSearchService(model_key="fast")


# ── Color Palettes ────────────────────────────────────────────────────────────

CLUSTER_COLORS = [
    '#FF2E93', '#0072FF', '#10B981', '#F59E0B',
    '#8B5CF6', '#EC4899', '#06B6D4', '#84CC16',
    '#F97316', '#14B8A6', '#E74C3C', '#3498DB'
]

DIMENSION_LABELS = {
    'cancer_type': '🩺 نوع السرطان',
    'treatment_type': '💊 نوع العلاج',
    'stage': '🔬 مرحلة السرطان',
    'gene': '🧬 الجينات/الطفرات',
    'demographics': '🧒 الفئة السكانية',
    'comorbidity': '❤️ الأمراض المصاحبة'
}


# ── Data Loading ──────────────────────────────────────────────────────────────

@st.cache_data
def get_dataset_with_tsne(
    sample_size: int = 1500,
    dimension: str = 'cancer_type',
    n_clusters: int = 6,
    method: str = 'tsne'
):
    """Load documents and create well-separated clusters with t-SNE/UMAP."""
    clusterer = get_clusterer()
    bert_service = get_bert()
    
    conn = sqlite3.connect(clusterer.db_path)
    cursor = conn.cursor()
    
    dim_map = {
        'cancer_type': 'cancer_type',
        'treatment_type': 'treatment_type',
        'stage': 'stage',
        'gene': 'gene',
        'demographics': 'demographics',
        'comorbidity': 'comorbidity'
    }
    dim_col = dim_map.get(dimension, 'cancer_type')
    
    # Get categories with sufficient documents
    cursor.execute(f"""
        SELECT {dim_col}, COUNT(*) as cnt
        FROM document_topics
        WHERE {dim_col} != 'other'
        GROUP BY {dim_col}
        HAVING cnt > 50
        ORDER BY cnt DESC
        LIMIT 10
    """)
    categories = [row[0] for row in cursor.fetchall()]
    
    if not categories:
        categories = ['breast', 'lung', 'colorectal', 'cardiovascular', 'neurological']
    
    # Sample evenly from each category
    docs = []
    per_cat = max(sample_size // len(categories), 30)
    
    for cat in categories:
        cursor.execute(f"""
            SELECT d.doc_id, d.text, t.{dim_col}
            FROM documents d
            JOIN document_topics t ON d.doc_id = t.doc_id
            WHERE t.{dim_col} = ?
            LIMIT ?
        """, (cat, per_cat))
        rows = cursor.fetchall()
        for row in rows:
            docs.append({
                'doc_id': row[0],
                'text': row[1] or '',
                'topic': row[2] or 'other'
            })
    
    conn.close()
    
    if not docs:
        return [], np.zeros((0, 2)), []
    
    # Get embeddings
    vectors = []
    valid_docs = []
    for doc in docs:
        vec = bert_service.get_vector(doc['doc_id'])
        if vec is not None:
            vectors.append(vec)
            valid_docs.append(doc)
    
    if not vectors:
        return [], np.zeros((0, 2)), []
    
    X = np.array(vectors, dtype=np.float32)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-Means clustering
    n_clusters = min(n_clusters, len(X_scaled) // 20, len(categories) + 2)
    n_clusters = max(n_clusters, 3)
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=20,
        max_iter=500,
        init='k-means++'
    )
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Dimensionality reduction
    if method == 'umap' and HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
        coords = reducer.fit_transform(X_scaled)
    else:
        # ✅ t-SNE with compatible parameters
        perplexity = min(30, len(X_scaled) // 3)
        
        # ✅ Handle different scikit-learn versions
        try:
            tsne = TSNE(
                n_components=2,
                random_state=42,
                perplexity=perplexity,
                early_exaggeration=12,
                learning_rate='auto',
                init='pca',
                max_iter=1000  # scikit-learn >= 1.2
            )
        except TypeError:
            tsne = TSNE(
                n_components=2,
                random_state=42,
                perplexity=perplexity,
                early_exaggeration=12,
                learning_rate='auto',
                init='pca',
                n_iter=1000  # scikit-learn < 1.2
            )
        coords = tsne.fit_transform(X_scaled)
    
    return valid_docs, coords, cluster_labels, categories


# ── Main App ─────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Topic Cluster Explorer",
        page_icon="🧬",
        layout="wide"
    )
    
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');
            html, body, [class*="css"], .stMarkdown, p, div, label {
                font-family: 'Cairo', sans-serif !important;
            }
            .main-title {
                text-align: center;
                font-size: 2.2rem;
                font-weight: 800;
                background: linear-gradient(135deg, #06B6D4, #3B82F6, #8B5CF6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.3rem;
            }
            .sub-title {
                text-align: center;
                color: #64748B;
                font-size: 1rem;
                margin-bottom: 1.5rem;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-title">🧬 مستكشف مجموعات الوثائق الطبية</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">تجميع دلالي مع مسافات واضحة باستخدام t-SNE</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ الإعدادات")
        
        dimension = st.selectbox(
            "البعد الطبي",
            options=list(DIMENSION_LABELS.keys()),
            format_func=lambda x: DIMENSION_LABELS[x]
        )
        
        reduction_method = st.selectbox(
            "طريقة تقليل الأبعاد",
            options=['tsne', 'umap' if HAS_UMAP else 'tsne'],
            format_func=lambda x: 't-SNE (مفضل)' if x == 'tsne' else 'UMAP (أسرع)'
        )
        
        n_clusters = st.slider("عدد المجموعات", 3, 10, 6)
        sample_size = st.slider("عدد الوثائق", 200, 2000, 1000)
        
        st.markdown("---")
        st.caption("📌 كل نقطة تمثل وثيقة طبية")
        st.caption("🎨 الألوان تمثل المجموعات المتشابهة")
        st.caption("🔍 مرر الفأرة فوق النقطة لعرض التفاصيل")
        st.caption("📊 t-SNE يحافظ على البنية المحلية للمجموعات")
    
    # Load data
    with st.spinner("جاري تحميل الوثائق وتطبيق التجميع..."):
        docs, coords, clusters, categories = get_dataset_with_tsne(
            sample_size=sample_size,
            dimension=dimension,
            n_clusters=n_clusters,
            method=reduction_method
        )
    
    if not docs or len(coords) == 0:
        st.error("لم يتم العثور على وثائق. تأكد من تشغيل precompute_topics.py أولاً.")
        return
    
    # Prepare data
    df = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'doc_id': [d['doc_id'] for d in docs],
        'topic': [d['topic'] for d in docs],
        'cluster': clusters,
        'text': [d['text'][:250] + '...' for d in docs]
    })
    
    # Calculate centroids for each cluster
    centroids = df.groupby('cluster')[['x', 'y']].mean().reset_index()
    
    # Build figure
    fig = go.Figure()
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster_id]
        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        
        # Get top topics in this cluster
        topic_counts = cluster_df['topic'].value_counts()
        top_topics = topic_counts.index[:2].tolist() if len(topic_counts) > 0 else ['other']
        top_topic_str = ' / '.join([t.upper() for t in top_topics])
        
        centroid = centroids[centroids['cluster'] == cluster_id]
        
        fig.add_trace(go.Scatter(
            x=cluster_df['x'],
            y=cluster_df['y'],
            mode='markers',
            marker=dict(
                size=9,
                color=color,
                opacity=0.85,
                line=dict(width=0.5, color='white')
            ),
            name=f"📁 {top_topic_str}",
            hovertext=cluster_df.apply(
                lambda row: (
                    f"<b>📄 معرف الوثيقة:</b> {row['doc_id']}<br>"
                    f"<b>🏷️ التصنيف:</b> {row['topic'].upper()}<br>"
                    f"<b>📋 المقتطف:</b><br>{row['text']}"
                ),
                axis=1
            ),
            hoverinfo='text'
        ))
    
    # Add centroids as separate trace
    fig.add_trace(go.Scatter(
        x=centroids['x'],
        y=centroids['y'],
        mode='markers',
        marker=dict(
            symbol='x',
            size=15,
            color='#0F172A',
            line=dict(width=3)
        ),
        name='🎯 مركز المجموعة',
        hoverinfo='skip'
    ))
    
    # Chart styling
    fig.update_layout(
        plot_bgcolor='rgba(248, 250, 252, 0.95)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=650,
        margin=dict(l=30, r=30, t=30, b=30),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(226, 232, 240, 0.9)',
            zeroline=True,
            zerolinecolor='rgba(148, 163, 184, 0.5)',
            showticklabels=False,
            range=[coords[:, 0].min() - 1, coords[:, 0].max() + 1]
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(226, 232, 240, 0.9)',
            zeroline=True,
            zerolinecolor='rgba(148, 163, 184, 0.5)',
            showticklabels=False,
            range=[coords[:, 1].min() - 1, coords[:, 1].max() + 1]
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#E2E8F0',
            borderwidth=1,
            font=dict(size=11, family='Cairo, sans-serif')
        ),
        hoverlabel=dict(
            bgcolor='#1E293B',
            font_size=13,
            font_color='#F8FAFC',
            font_family='Cairo, sans-serif'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.markdown("---")
    st.subheader("📊 إحصائيات المجموعات")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        cluster_stats = []
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_df = df[df['cluster'] == cluster_id]
            topic_counts = cluster_df['topic'].value_counts()
            top_topics = ', '.join([f"{t.upper()} ({c})" for t, c in topic_counts.head(3).items()])
            cluster_stats.append({
                'المجموعة': f'{cluster_id + 1}',
                'عدد الوثائق': len(cluster_df),
                'أهم التصنيفات': top_topics
            })
        
        st.dataframe(
            pd.DataFrame(cluster_stats),
            column_config={
                'المجموعة': st.column_config.TextColumn('المجموعة'),
                'عدد الوثائق': st.column_config.NumberColumn('عدد الوثائق'),
                'أهم التصنيفات': st.column_config.TextColumn('أهم التصنيفات')
            },
            hide_index=True,
            use_container_width=True
        )
    
    with col2:
        # Topic distribution pie chart
        import plotly.express as px
        topic_counts = df['topic'].value_counts().reset_index()
        topic_counts.columns = ['Topic', 'Count']
        
        fig_pie = px.pie(
            topic_counts.head(8),
            values='Count',
            names='Topic',
            hole=0.35,
            title='توزيع التصنيفات',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)


if __name__ == '__main__':
    main()