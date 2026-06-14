# ui/app.py
"""
IR System 2026 — Streamlit UI
Supports: VSM TF-IDF | BM25 | Simple TF-IDF | BERT Embeddings
"""

import streamlit as st
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.retrieval.search_service import SearchService
from services.retrieval.vsm_search_service import VSMSearchService
from services.retrieval.bm25_search_service import BM25SearchService
from services.ranking.embeddings.bert_search_service import BERTSearchService


# ── Cached service loaders (only initialized once per session) ────────────────

@st.cache_resource
def get_simple_service():
    """Simple TF-IDF baseline service."""
    return SearchService()


@st.cache_resource
def get_vsm_service():
    """VSM TF-IDF with cosine similarity."""
    return VSMSearchService()


@st.cache_resource
def get_bm25_service():
    """BM25 probabilistic ranking service."""
    return BM25SearchService(k1=1.5, b=0.75)


@st.cache_resource
def get_bert_service():
    """BERT semantic search service."""
    return BERTSearchService(model_key="fast")


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="IR System 2026",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 Information Retrieval System 2026")
    st.markdown("---")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        model = st.selectbox(
            "📊 Retrieval Model",
            [
                "BM25 (Best Matching 25)",
                "VSM TF-IDF (Cosine Similarity)",
                "BERT Embeddings (Semantic Search)",
                "Simple TF-IDF (Baseline)",
            ],
            help=(
                "**BM25** — Advanced probabilistic ranking (Recommended)\n\n"
                "**VSM TF-IDF** — Classic vector space model with cosine similarity\n\n"
                "**BERT Embeddings** — Semantic search using Sentence-BERT + FAISS\n\n"
                "**Simple TF-IDF** — Baseline inverted index with TF scoring"
            )
        )

        # BM25 parameters (only show when BM25 is selected)
        if "BM25" in model:
            st.markdown("---")
            st.subheader("🎯 BM25 Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                k1 = st.slider(
                    "k1 (TF Saturation)", 
                    min_value=0.5, 
                    max_value=2.5, 
                    value=1.5, 
                    step=0.1,
                    help="Higher = more influence from term frequency"
                )
            with col2:
                b = st.slider(
                    "b (Length Normalization)", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=0.75, 
                    step=0.05,
                    help="0 = no normalization, 1 = full normalization"
                )
            
            st.caption(f"BM25 Formula: Σ IDF(q) × ((tf × (k1+1)) / (tf + k1 × (1-b + b×|D|/avgDL)))")
            st.caption(f"Current: k1={k1}, b={b}")

        top_k = st.slider("📄 Number of results", 5, 50, 10)

        st.markdown("---")
        st.subheader("🎨 Display Options")
        show_text      = st.checkbox("📖 Show document text",    value=True)
        max_text_len   = st.slider("📏 Max text length", 100, 1000, 300)
        show_matching  = st.checkbox("🔍 Show matching details",  value=True)
        show_score_exp = st.checkbox("📐 Show score explanation", value=False)

        st.markdown("---")

        # Stats panel
        try:
            if "BM25" in model:
                svc = get_bm25_service()
                stats = svc.get_stats()
                st.info(
                    f"📊 **BM25 Stats**\n\n"
                    f"🔤 Unique terms: {stats['unique_terms']:,}\n"
                    f"📄 Documents: {stats['total_documents']:,}\n"
                    f"📐 Avg length: {stats['avg_doc_length']:.1f}\n"
                    f"🎯 k1={svc.k1}, b={svc.b}"
                )

            elif "VSM" in model:
                svc = get_vsm_service()
                st.info(
                    f"📊 **VSM Stats**\n\n"
                    f"🔤 Unique terms: {len(svc.inverted_index):,}\n"
                    f"📄 Documents: {svc.total_docs:,}\n"
                    f"🎯 Model: Cosine Similarity"
                )

            elif "BERT" in model:
                svc = get_bert_service()
                stats = svc.get_stats()
                st.info(
                    f"📊 **BERT Stats**\n\n"
                    f"🤖 Model: {stats['model_name']}\n"
                    f"📐 Vector dim: {stats['vector_dim']}\n"
                    f"📄 Documents: {stats['total_documents']:,}\n"
                    f"🎯 Search: FAISS Inner Product"
                )

            else:
                svc = get_simple_service()
                st.info(
                    f"📊 **Simple TF-IDF Stats**\n\n"
                    f"🔤 Terms in index: {len(svc.index):,}\n"
                    f"📄 Documents: {len(svc.documents):,}\n"
                    f"🎯 Method: TF Score Sum"
                )
        except Exception as e:
            st.error(f"Error loading service: {e}")

    # ── Query Input ───────────────────────────────────────────────────────────
    st.header("📝 Enter your search query")

    example_queries = [
        "cloud storage backup",
        "machine learning algorithms",
        "data recovery solutions",
        "artificial intelligence applications",
    ]

    cols = st.columns(len(example_queries))
    for i, example in enumerate(example_queries):
        with cols[i]:
            if st.button(f"🔍 {example}", key=f"btn_{i}"):
                st.session_state.query = example

    query = st.text_input(
        "Search query:",
        value=st.session_state.get("query", ""),
        placeholder="e.g., cloud storage backup solutions...",
        label_visibility="collapsed"
    )

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)

    # ── Search & Results ──────────────────────────────────────────────────────
    if search_button and query:
        start_time = time.time()

        with st.spinner("🔍 Searching..."):
            try:
                results     = []
                method_used = model

                if "BM25" in model:
                    svc = get_bm25_service()
                    # Update parameters if changed
                    if 'k1' in locals():
                        svc.k1 = k1
                        svc.b = b
                    results = svc.search(query, top_k=top_k)
                    method_used = f"BM25 (k1={svc.k1}, b={svc.b})"

                elif "VSM" in model:
                    results = get_vsm_service().search(query, top_k=top_k)
                    method_used = "VSM TF-IDF (Cosine Similarity)"

                elif "BERT" in model:
                    results = get_bert_service().search(query, top_k=top_k)
                    method_used = "BERT Semantic Search (FAISS)"

                else:
                    results = get_simple_service().search(query, top_k=top_k)
                    method_used = "Simple TF-IDF (TF Scoring)"

            except Exception as e:
                st.error(f"Error during search: {e}")
                results = []

        search_time = time.time() - start_time

        if results:
            st.success(f"✅ Found {len(results)} results in {search_time:.3f} seconds")
            st.caption(f"📐 Method: {method_used}")
            st.markdown("---")

            for i, result in enumerate(results, 1):
                with st.container():
                    st.markdown(f"### {i}. 📄 Document `{result['doc_id']}`")
                    
                    # Score with color coding
                    score = result['score']
                    if score > 0.7:
                        score_color = "🟢"
                    elif score > 0.4:
                        score_color = "🟡"
                    else:
                        score_color = "🟠"
                    
                    st.markdown(f"**🎯 Relevance Score:** `{score_color} {score:.6f}`")

                    if show_score_exp and result.get("method"):
                        st.caption(f"📐 Scoring method: {result.get('method', method_used)}")

                    if show_text and result.get("text"):
                        st.markdown("**📖 Document Content:**")
                        preview = result["text"][:max_text_len]
                        if len(result["text"]) > max_text_len:
                            preview += "..."
                        st.markdown(f"> {preview}")

                        with st.expander("📚 Show full document"):
                            st.write(result.get("full_text", result["text"]))

                    if show_matching:
                        with st.expander("🔍 Show matching terms and scoring details"):
                            _show_matching_details(result, query, model)

                    st.markdown("---")

        else:
            st.warning("⚠️ No results found. Try different keywords.")
            st.info(
                "💡 **Search Tips:**\n"
                "- Try more specific keywords\n"
                "- Use synonyms\n"
                "- For BM25: Run `python scripts/build_bm25_index.py` first\n"
                "- For BERT: Run `python scripts/build_bert_index.py` first"
            )


# ── Helper: matching details panel ────────────────────────────────────────────

def _show_matching_details(result: dict, query: str, model: str) -> None:
    """Show per-model matching details inside an expander."""
    try:
        import streamlit as st
        if model == "BERT Embeddings (Semantic Search)":
            svc    = get_bert_service()
            tokens = svc.preprocessor.process(query)
            st.write(f"**Query tokens:** `{tokens}`")
            st.write(f"**Score type:** Cosine similarity (BERT vector space)")
            st.write(f"**Score:** `{result['score']:.6f}` — closer to 1.0 = more similar")

        elif "VSM" in model:
            svc    = get_vsm_service()
            tokens = svc.query_processor.process_query(query)
            st.write(f"**Query tokens:** `{tokens}`")

            for token in tokens:
                details = svc.get_term_details(result["doc_id"], token)
                if details["tf"] > 0 or details["idf"] > 0:
                    st.write(f"\n**`{token}`:**")
                    st.write(f"- TF = `{details['tf']:.4f}`")
                    st.write(f"- IDF = `{details['idf']:.4f}`")
                    st.write(f"- TF-IDF = `{details['tfidf']:.4f}`")
                else:
                    st.write(f"**`{token}`:** not found in this document")

        elif "BM25" in model:
            svc = get_bm25_service()
            tokens = svc.query_processor.process_query(query)
            st.write(f"**Query tokens:** `{tokens}`")
            st.write(f"**Parameters:** k1={svc.k1}, b={svc.b}")
            avg_doc_len = svc.document_store.avg_doc_length
            doc_len = svc.document_store.get_length(result["doc_id"])
            st.write(f"Document length: `{doc_len}`, Average length: `{avg_doc_len:.1f}`")
            
            for token in tokens:
                tf = svc.inverted_index.get_term_frequency(token, result["doc_id"])
                df = svc.inverted_index.doc_frequency.get(token, 0)
                idf = svc.scorer.compute_idf(df, svc.document_store.total_docs)
                score = svc.scorer.score_term(tf, doc_len, avg_doc_len, idf)
                
                if score > 0:
                    st.write(f"\n**`{token}`:**")
                    st.write(f"- TF = `{tf}`")
                    st.write(f"- IDF = `{idf:.4f}`")
                    st.write(f"- BM25 Term Score = `{score:.4f}`")
                else:
                    st.write(f"**`{token}`:** tf = 0 (no match)")

        else:
            svc    = get_simple_service()
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
                st.write(f"- `{token}`: appears {freq} time(s)")

    except Exception as e:
        st.write(f"Error showing details: {e}")


if __name__ == "__main__":
    main()


