# ui/app.py
"""
IR System 2026 — Streamlit UI
Supports: VSM TF-IDF | Simple TF-IDF | BERT Embeddings
"""

import streamlit as st
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.retrieval.search_service import SearchService
from services.ranking.vsm_TIf_DF.vsm_tfidf_custom import VSM_TFIDF_Custom
from services.ranking.embeddings.bert_search_service import BERTSearchService


# ── Cached service loaders (only initialized once per session) ────────────────

@st.cache_resource
def get_simple_service():
    return SearchService()


@st.cache_resource
def get_vsm_service():
    return VSM_TFIDF_Custom()


@st.cache_resource
def get_bert_service():
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
                "VSM_TF-IDF",
                "BERT Embeddings",
                "Simple TF-IDF (Baseline)",
            ],
            help=(
                "**VSM TF-IDF** — Classic vector space model with cosine similarity\n\n"
                "**BERT Embeddings** — Semantic search using Sentence-BERT + FAISS\n\n"
                "**Simple TF-IDF** — Baseline inverted index with TF scoring"
            )
        )

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
            if model == "VSM_TF-IDF":
                svc = get_vsm_service()
                st.info(
                    f"📊 **System Stats**\n\n"
                    f"🔤 Unique terms: {len(svc.inverted_index):,}\n"
                    f"📄 Documents: {svc.total_docs:,}\n"
                    f"🎯 Model: VSM + Cosine Similarity"
                )

            elif model == "BERT Embeddings":
                svc = get_bert_service()
                stats = svc.get_stats()
                st.info(
                    f"📊 **System Stats**\n\n"
                    f"🤖 Model: {stats['model_name']}\n"
                    f"📐 Vector dim: {stats['vector_dim']}\n"
                    f"📄 Documents: {stats['total_documents']:,}\n"
                    f"🎯 Search: FAISS Inner Product"
                )

            else:
                svc = get_simple_service()
                st.info(
                    f"📊 **System Stats**\n\n"
                    f"🔤 Terms in index: {len(svc.index):,}\n"
                    f"📄 Documents: {len(svc.documents):,}\n"
                    f"🎯 Active model: {model}"
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

                if model == "VSM_TF-IDF":
                    results     = get_vsm_service().search(query, top_k=top_k)
                    method_used = "VSM TF-IDF (Cosine Similarity)"

                elif model == "BERT Embeddings":
                    results     = get_bert_service().search(query, top_k=top_k)
                    method_used = f"BERT Semantic Search (FAISS)"

                else:
                    results     = get_simple_service().search(query, top_k=top_k)
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
                st.markdown(f"### {i}. 📄 Document `{result['doc_id']}`")
                st.markdown(f"**🎯 Relevance Score:** `{result['score']:.4f}`")

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
                    with st.expander("🔍 Show matching details"):
                        _show_matching_details(result, query, model)

                st.markdown("---")

        else:
            st.warning("⚠️ No results found. Try different keywords.")
            st.info(
                "💡 **Search Tips:**\n"
                "- Try more specific keywords\n"
                "- Use synonyms\n"
                "- Run `python scripts/build_index.py` and `python scripts/build_bert_index.py` first"
            )

    # ── About Section ─────────────────────────────────────────────────────────
    with st.expander("ℹ️ About this system"):
        st.markdown("""
        ### 📚 Information Retrieval System 2026

        **Dataset:** MS MARCO Passage
        - Testing mode: 5,000 documents
        - Full mode: 200,000 documents

        **Retrieval Models:**

        | Model | Description | Similarity |
        |-------|-------------|------------|
        | **VSM TF-IDF** | Vector Space Model with TF-IDF | Cosine Similarity |
        | **BERT Embeddings** | Sentence-BERT dense vectors + FAISS | Inner Product (≈ Cosine) |
        | **Simple TF-IDF** | Inverted index with TF scoring | TF Score Sum |

        **Formulas:**
        - **TF(t,d)** = count(t,d) / len(d)
        - **IDF(t)** = log₁₀((N+1)/(df(t)+1)) + 1
        - **TF-IDF(t,d)** = TF × IDF
        - **BERT cos(q,d)** = q · d  (L2-normalized dot product)

        **Technologies:** Python · Streamlit · scikit-learn · sentence-transformers · FAISS
        """)


# ── Helper: matching details panel ────────────────────────────────────────────

def _show_matching_details(result: dict, query: str, model: str) -> None:
    """Show per-model matching details inside an expander."""
    try:
        if model == "BERT Embeddings":
            svc    = get_bert_service()
            tokens = svc.preprocessor.process(query)
            st.write(f"**Query tokens:** `{tokens}`")
            st.write(f"**Score type:** Cosine similarity (BERT vector space)")
            st.write(f"**Score:** `{result['score']:.6f}` — closer to 1.0 = more similar")

        elif model == "VSM_TF-IDF":
            svc    = get_vsm_service()
            tokens = svc.preprocessor.process(query)
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

        else:
            svc    = get_simple_service()
            tokens = svc.preprocessor.process(query)
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