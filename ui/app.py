# ui/app.py
import streamlit as st
import sys
import os
import time

# Add path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.retrieval.search_service import SearchService
from services.ranking.vsm_TIf_DF.vsm_tfidf_custom import VSM_TFIDF_Custom


@st.cache_resource
def get_simple_service():
    return SearchService()


@st.cache_resource
def get_vsm_service():
    return VSM_TFIDF_Custom()


def main():
    st.set_page_config(
        page_title="IR System 2026",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Information Retrieval System 2026")
    st.markdown("---")
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        model = st.selectbox(
            "📊 Retrieval Model",
            ["VSM_TF-IDF", "Simple TF-IDF (Baseline)"],
            help="Select the ranking algorithm\n\n"
                 "- VSM_TF-IDF: Vector Space Model with TF-IDF and Cosine Similarity\n"
                 "- Simple TF-IDF: Baseline inverted index with TF scoring"
        )
        
        top_k = st.slider("📄 Number of results", 5, 50, 10)
        
        st.markdown("---")
        
        st.subheader("🎨 Display Options")
        show_text = st.checkbox("📖 Show document text", value=True)
        max_text_length = st.slider("📏 Max text length", 100, 1000, 300)
        show_matching = st.checkbox("🔍 Show matching terms", value=True)
        show_explanation = st.checkbox("📐 Show score explanation", value=False)
        
        st.markdown("---")
        
        try:
            if model == "VSM_TF-IDF":
                service = get_vsm_service()
                st.info(f"📊 **System Stats**\n\n"
                       f"🔤 Unique terms: {len(service.inverted_index):,}\n"
                       f"📄 Documents: {service.total_docs:,}\n"
                       f"🎯 Model: VSM with Cosine Similarity")
            else:
                service = get_simple_service()
                st.info(f"📊 **System Stats**\n\n"
                       f"🔤 Terms in index: {len(service.index):,}\n"
                       f"📄 Documents: {len(service.documents):,}\n"
                       f"🎯 Active model: {model}")
        except Exception as e:
            st.error(f"Error loading service: {e}")
    
    st.header("📝 Enter your search query")
    
    example_queries = ["cloud storage backup", "machine learning algorithms", "data recovery solutions", "artificial intelligence applications"]
    cols = st.columns(len(example_queries))
    for i, example in enumerate(example_queries):
        with cols[i]:
            if st.button(f"🔍 {example}", key=f"btn_{i}"):
                st.session_state.query = example
    
    query = st.text_input(
        "Search query:",
        value=st.session_state.get("query", ""),
        placeholder="e.g., cloud storage backup solutions for enterprise...",
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    if search_button and query:
        start_time = time.time()
        
        with st.spinner("🔍 Searching..."):
            try:
                results = []
                method_used = model
                
                if model == "VSM_TF-IDF":
                    vsm_service = get_vsm_service()
                    results = vsm_service.search(query, top_k=top_k)
                    method_used = "VSM_TF-IDF (Cosine Similarity)"
                else:
                    simple_service = get_simple_service()
                    results = simple_service.search(query, top_k=top_k)
                    method_used = "Simple TF-IDF (TF Scoring)"
                    
            except Exception as e:
                st.error(f"Error during search: {e}")
                results = []
        
        search_time = time.time() - start_time
        
        if results:
            st.success(f"✅ Found {len(results)} results in {search_time:.3f} seconds")
            st.caption(f"📐 Method used: {method_used}")
            st.markdown("---")
            
            for i, result in enumerate(results, 1):
                st.markdown(f"### {i}. 📄 Document `{result['doc_id']}`")
                st.markdown(f"**🎯 Relevance Score:** `{result['score']:.4f}`")
                
                if show_explanation and result.get('method'):
                    st.caption(f"📐 Scoring method: {result.get('method', method_used)}")
                
                if show_text and result.get('text'):
                    st.markdown("**📖 Document Content:**")
                    text_preview = result['text'][:max_text_length]
                    if len(result['text']) > max_text_length:
                        text_preview += "..."
                    
                    st.markdown(f"> {text_preview}")
                    
                    with st.expander("📚 Show full document"):
                        st.write(result.get('full_text', result['text']))
                
                if show_matching:
                    with st.expander("🔍 Show matching terms and scoring details"):
                        try:
                            if model == "VSM_TF-IDF":
                                vsm_service = get_vsm_service()
                                tokens = vsm_service.preprocessor.process(query)
                                
                                if tokens:
                                    st.write("**Query terms analysis:**")
                                    st.write(f"Tokens after preprocessing: `{tokens}`")
                                    
                                    st.write("\n**Term weights in this document:**")
                                    
                                    for token in tokens:
                                        term_details = vsm_service.get_term_details(result['doc_id'], token)
                                        
                                        if term_details['tf'] > 0 or term_details['idf'] > 0:
                                            st.write(f"\n**`{token}`:**")
                                            st.write(f"- **TF** = {term_details['tf']:.4f}")
                                            st.write(f"- **IDF** = {term_details['idf']:.4f}")
                                            st.write(f"- **TF-IDF** = {term_details['tfidf']:.4f}")
                                        else:
                                            st.write(f"\n**`{token}`:** not found in this document")
                            
                            else:
                                simple_service = get_simple_service()
                                tokens = simple_service.preprocessor.process(query)
                                
                                if tokens:
                                    st.write("**Query terms analysis:**")
                                    st.write(f"Tokens after preprocessing: `{tokens}`")
                                    
                                    st.write("\n**Term frequency in this document:**")
                                    for token in tokens:
                                        freq = 0
                                        if token in simple_service.index:
                                            term_data = simple_service.index[token]
                                            if isinstance(term_data, dict):
                                                freq = term_data.get(result['doc_id'], 0)
                                            elif isinstance(term_data, list):
                                                for d, f in term_data:
                                                    if str(d) == result['doc_id']:
                                                        freq = f
                                                        break
                                        st.write(f"- `{token}`: appears {freq} time(s)")
                            
                        except Exception as e:
                            st.write(f"Error showing matching terms: {e}")
                
                st.markdown("---")
        else:
            st.warning("⚠️ No results found. Try different keywords.")
            
            st.info("💡 **Search Tips:**\n"
                   "- Try using more specific keywords\n"
                   "- Use synonyms of your search terms\n"
                   "- Make sure the index is loaded correctly\n"
                   "- Try running `python scripts/build_index.py` first")
    
    with st.expander("ℹ️ About this system - Detailed Information"):
        st.markdown("""
        ### 📚 Information Retrieval System 2026
        
        **Dataset:** MS MARCO Passage
        - Testing mode: 5,000 documents
        - Full mode: 200,000 documents
        
        **Processing Pipeline:**
        1. 🔤 Lowercasing
        2. 🔢 Number removal
        3. ✂️ Punctuation removal
        4. 🚫 Stopword filtering
        5. ✨ Tokenization
        
        **Retrieval Models:**
        
        | Model | Description | Formula |
        |-------|-------------|---------|
        | **VSM_TF-IDF** | Vector Space Model with TF-IDF and Cosine Similarity | `cos(q,d) = (Σ tfidf(q) × tfidf(d)) / (||q|| × ||d||)` |
        | **Simple TF-IDF** | Baseline inverted index with TF scoring | `score = Σ TF(t,d)` |
        
        **Formulas:**
        - **TF(t,d)** = count(t,d) / len(d)
        - **IDF(t)** = log10((N+1)/(df(t)+1)) + 1
        - **TF-IDF(t,d)** = TF(t,d) × IDF(t)
        
        **Evaluation Metrics:** MAP, nDCG@10, P@10, Recall
        
        **Technologies:** Python, Streamlit, scikit-learn, NLTK
        """)

if __name__ == "__main__":
    main()