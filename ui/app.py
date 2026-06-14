# ui/app.py
"""
IR System 2026 — Streamlit UI
Models: BM25 | VSM TF-IDF | BERT Embeddings | Hybrid Serial | Hybrid Parallel | Simple TF-IDF
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
from services.ranking.hybrid.hybrid_search_service import HybridSearchService


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
    # Share the already-loaded BM25 and BERT services to avoid double loading
    return HybridSearchService(
        bm25_service=get_bm25_service(),
        bert_service=get_bert_service(),
    )


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

        # ── BM25 parameters (shown for BM25 and both Hybrid modes) ───────────
        show_bm25_params = any(x in model for x in ["BM25", "Hybrid"])
        if show_bm25_params:
            st.markdown("---")
            st.subheader("🎯 BM25 Parameters")
            col1, col2 = st.columns(2)
            with col1:
                k1 = st.slider("k1 (TF Saturation)", 0.5, 2.5, 1.5, 0.1,
                                help="Controls how quickly TF saturates. Higher = more influence from repeated terms.")
            with col2:
                b = st.slider("b (Length Norm)", 0.0, 1.0, 0.75, 0.05,
                               help="0 = no length penalty, 1 = full normalization.")

            st.caption(f"Formula: IDF × (tf × (k1+1)) / (tf + k1 × (1 - b + b × |D|/avgDL))")
            st.caption(f"Current: **k1 = {k1}**, **b = {b}**")

        # ── Hybrid-specific options ───────────────────────────────────────────
        if "Hybrid Parallel" in model:
            st.markdown("---")
            st.subheader("🔀 Fusion Settings")
            fusion = st.radio("Fusion method", ["rrf", "linear"],
                              format_func=lambda x: "RRF (Reciprocal Rank)" if x == "rrf" else "Linear (Weighted Sum)")

            if fusion == "linear":
                bm25_w = st.slider("BM25 weight", 0.0, 1.0, 0.4, 0.05)
                bert_w = st.slider("BERT weight", 0.0, 1.0, 0.6, 0.05)
                st.caption(f"Combined weight: {bm25_w + bert_w:.2f} (ideally = 1.0)")
            else:
                bm25_w, bert_w = 0.4, 0.6   # unused for RRF but needed as defaults

        if "Hybrid Serial" in model:
            st.markdown("---")
            st.subheader("🔀 Serial Settings")
            bm25_candidates = st.slider("BM25 candidates for rerank", 20, 200, 100, 10,
                                         help="How many BM25 results BERT will re-rank.")

        top_k = st.slider("📄 Number of results", 5, 50, 10)

        st.markdown("---")
        st.subheader("🎨 Display")
        show_text      = st.checkbox("📖 Show document text", value=True)
        max_text_len   = st.slider("📏 Max text length", 100, 1000, 300)
        show_matching  = st.checkbox("🔍 Show matching details", value=True)
        show_score_exp = st.checkbox("📐 Show score explanation", value=False)

        st.markdown("---")
        _render_stats(model)

    # ── Query area ────────────────────────────────────────────────────────────
    st.header("📝 Enter your search query")

    examples = ["cloud storage backup", "machine learning algorithms",
                "data recovery solutions", "artificial intelligence applications"]
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        with cols[i]:
            if st.button(f"🔍 {ex}", key=f"ex_{i}"):
                st.session_state.query = ex

    query = st.text_input("Search query:", value=st.session_state.get("query", ""),
                          placeholder="e.g., cloud storage backup solutions...",
                          label_visibility="collapsed")

    c1, c2, c3 = st.columns([1, 1, 4])
    with c1:
        search_btn = st.button("🔍 Search", type="primary", use_container_width=True)

    # ── Search ────────────────────────────────────────────────────────────────
    if search_btn and query:
        t0 = time.time()
        results = []
        method_used = model

        with st.spinner("🔍 Searching..."):
            try:
                if "BM25" in model and "Hybrid" not in model:
                    svc = get_bm25_service()
                    svc.k1 = k1
                    svc.b  = b
                    results = svc.search(query, top_k=top_k)
                    method_used = f"BM25 (k1={k1}, b={b})"

                elif "Hybrid Serial" in model:
                    svc = get_hybrid_service()
                    svc.k1 = k1
                    svc.b  = b
                    cands = bm25_candidates if "bm25_candidates" in dir() else 100
                    results = svc.search(query, mode="serial", top_k=top_k, bm25_candidates=cands)
                    method_used = f"Hybrid Serial — BM25(k1={k1},b={b}) → BERT rerank"

                elif "Hybrid Parallel" in model:
                    svc = get_hybrid_service()
                    svc.k1 = k1
                    svc.b  = b
                    fus   = fusion if "fusion" in dir() else "rrf"
                    bw    = bm25_w if "bm25_w" in dir() else 0.4
                    ew    = bert_w if "bert_w" in dir() else 0.6
                    results = svc.search(query, mode="parallel", fusion=fus,
                                         top_k=top_k, bm25_weight=bw, bert_weight=ew)
                    method_used = f"Hybrid Parallel {fus.upper()} — BM25(k1={k1},b={b}) + BERT"

                elif "VSM" in model:
                    results = get_vsm_service().search(query, top_k=top_k)
                    method_used = "VSM TF-IDF (Cosine Similarity)"

                elif "BERT" in model:
                    results = get_bert_service().search(query, top_k=top_k)
                    method_used = "BERT Semantic Search (FAISS)"

                else:
                    results = get_simple_service().search(query, top_k=top_k)
                    method_used = "Simple TF-IDF"

            except Exception as e:
                st.error(f"Search error: {e}")

        elapsed = time.time() - t0

        if results:
            st.success(f"✅ {len(results)} results in {elapsed:.3f}s")
            st.caption(f"📐 Method: {method_used}")
            st.markdown("---")

            for i, r in enumerate(results, 1):
                score = r["score"]
                dot   = "🟢" if score > 0.7 else ("🟡" if score > 0.3 else "🟠")

                st.markdown(f"### {i}. 📄 `{r['doc_id']}`")
                st.markdown(f"**🎯 Score:** `{dot} {score:.6f}`")

                # Show hybrid rank info if available
                if "bm25_rank" in r:
                    st.caption(f"BM25 rank: {r['bm25_rank']} | BERT rank: {r['bert_rank']}")

                if show_score_exp and r.get("method"):
                    st.caption(f"Method: {r['method']}")

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
                        _show_matching(r, query, model)

                st.markdown("---")
        else:
            st.warning("⚠️ No results found.")

    # ── About ─────────────────────────────────────────────────────────────────
    with st.expander("ℹ️ About this system"):
        st.markdown("""
        ### IR System 2026

        | Model | Description | Scoring |
        |-------|-------------|---------|
        | **BM25** | Probabilistic, term saturation | BM25 formula |
        | **Hybrid Serial** | BM25 candidates → BERT rerank | BERT cosine |
        | **Hybrid Parallel RRF** | BM25 + BERT fused (rank-based) | 1/(k+rank) |
        | **Hybrid Parallel Linear** | BM25 + BERT fused (score-based) | w1×BM25 + w2×BERT |
        | **VSM TF-IDF** | Vector Space Model | Cosine similarity |
        | **BERT** | Semantic dense retrieval | Inner product (FAISS) |
        | **Simple TF-IDF** | Baseline inverted index | TF sum |

        **BM25 Parameters:**
        - **k1** — controls term frequency saturation (typical: 1.2–2.0)
        - **b** — length normalization (0 = off, 0.75 = standard, 1 = full)

        **Hybrid Fusion:**
        - **RRF** — rank-based, no score normalization needed, very robust
        - **Linear** — weighted sum of min-max normalized scores, interpretable
        """)


# ── Sidebar stats ─────────────────────────────────────────────────────────────

def _render_stats(model: str):
    try:
        if "Hybrid" in model:
            svc   = get_hybrid_service()
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
            svc   = get_bm25_service()
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
            svc   = get_bert_service()
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
            svc    = get_bert_service()
            tokens = svc.preprocessor.process(query)
            st.write(f"**Tokens:** `{tokens}`")
            st.write(f"**Score:** `{result['score']:.6f}` (cosine similarity, 1.0 = identical)")

        elif "Hybrid Serial" in model:
            svc    = get_hybrid_service()
            tokens = svc.bm25.query_processor.process_query(query)
            st.write(f"**Query tokens:** `{tokens}`")
            st.write("**Stage 1:** BM25 retrieved candidates")
            st.write("**Stage 2:** BERT re-ranked by semantic similarity")
            st.write(f"**Final score:** `{result['score']:.6f}` (BERT cosine)")

        elif "Hybrid Parallel" in model:
            svc = get_hybrid_service()
            tokens = svc.bm25.query_processor.process_query(query)
            st.write(f"**Query tokens:** `{tokens}`")
            st.write(f"**BM25 rank:** `{result.get('bm25_rank', '-')}`")
            st.write(f"**BERT rank:** `{result.get('bert_rank', '-')}`")
            st.write(f"**Fused score:** `{result['score']:.6f}`")

        elif "VSM" in model:
            svc    = get_vsm_service()
            tokens = svc.query_processor.process_query(query)
            st.write(f"**Query tokens:** `{tokens}`")
            for token in tokens:
                d = svc.get_term_details(result["doc_id"], token)
                if d["tf"] > 0 or d["idf"] > 0:
                    st.write(f"**`{token}`:** TF={d['tf']:.4f} | IDF={d['idf']:.4f} | TF-IDF={d['tfidf']:.4f}")
                else:
                    st.write(f"**`{token}`:** not found in document")

        elif "BM25" in model:
            svc    = get_bm25_service()
            tokens = svc.query_processor.process_query(query)
            st.write(f"**Query tokens:** `{tokens}`")
            st.write(f"**k1={svc.k1}, b={svc.b}**")
            avg = svc.document_store.avg_doc_length
            dlen = svc.document_store.get_length(result["doc_id"])
            st.write(f"Doc length: `{dlen}` | Avg: `{avg:.1f}`")
            for token in tokens:
                tf   = svc.inverted_index.get_term_frequency(token, result["doc_id"])
                df   = svc.inverted_index.doc_frequency.get(token, 0)
                idf  = svc.scorer.compute_idf(df, svc.document_store.total_docs)
                sc   = svc.scorer.score_term(tf, dlen, avg, idf)
                if sc > 0:
                    st.write(f"**`{token}`:** TF={tf} | IDF={idf:.4f} | BM25={sc:.4f}")
                else:
                    st.write(f"**`{token}`:** not found")

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
                st.write(f"- `{token}`: {freq} occurrence(s)")

    except Exception as e:
        st.write(f"Error: {e}")


if __name__ == "__main__":
    main()