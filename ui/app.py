import streamlit as st
import sys
import os
import time

# إضافة المسار
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.retrieval.search_service import SearchService

# تهيئة خدمة البحث
@st.cache_resource
def get_search_service():
  return SearchService()

def main():
    st.title("🔍 Information Retrieval System 2026")
    st.markdown("---")
    
    # شريط جانبي للإعدادات
    with st.sidebar:
        st.header("⚙️ Settings")
        
        model = st.selectbox(
            "Retrieval Model",
            ["Simple TF-IDF", "BM25", "Hybrid Serial", "Hybrid Parallel"],
            help="Select the ranking algorithm"
        )
        
        top_k = st.slider("Number of results", 5, 50, 10)
        
        st.markdown("---")
        
        # خيارات العرض
        st.subheader("Display Options")
        show_text = st.checkbox("📄 Show document text", value=True)
        max_text_length = st.slider("Max text length", 100, 1000, 300)
        
        st.markdown("---")
        
        # إحصائيات
        service = get_search_service()
        st.info(f"📊 **Index Stats**\n\n"
                f"- Terms: {len(service.index)}\n"
                f"- Documents: {len(service.documents)}\n"
                f"- Model: {model}")
    
    # منطقة البحث الرئيسية
    st.header("📝 Enter your query")
    query = st.text_input("Search query:", placeholder="e.g., cloud storage backup solutions...")
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # نتائج البحث
    if search_button and query:
        start_time = time.time()
        
        with st.spinner("Searching..."):
            service = get_search_service()
            results = service.search(query, top_k=top_k)
        
        search_time = time.time() - start_time
        
        if results:
            st.success(f"✅ Found {len(results)} results in {search_time:.3f} seconds")
            st.markdown("---")
            
            # عرض النتائج
            for i, result in enumerate(results, 1):
                # عنوان النتيجة
                st.markdown(f"### {i}. 📄 Document `{result['doc_id']}`")
                st.markdown(f"**Relevance Score:** `{result['score']:.2f}`")
                
                # عرض النص إذا كان متاحاً ومطلوباً
                if show_text and result.get('text'):
                    st.markdown("**Document Content:**")
                    text_preview = result['text'][:max_text_length]
                    if len(result['text']) > max_text_length:
                        text_preview += "..."
                    st.markdown(f"> {text_preview}")
                    
                    # زر لتوسيع النص
                    with st.expander("📖 Show full document"):
                        st.write(result.get('full_text', result['text']))
                
                # عرض الكلمات المفتاحية التي ظهرت
                with st.expander("🔍 Show matching terms"):
                    tokens = service.preprocessor.process(query)
                    st.write("**Query terms found in this document:**")
                    doc_matches = []
                    for token in tokens:
                        if token in service.index and result['doc_id'] in service.index[token]:
                            freq = service.index[token][result['doc_id']]
                            doc_matches.append(f"- `{token}`: appears {freq} time(s)")
                    if doc_matches:
                        for match in doc_matches:
                            st.write(match)
                    else:
                        st.write("No direct term matches (score from other terms)")
                
                st.markdown("---")
        else:
            st.warning("⚠️ No results found. Try different keywords.")
            
            # اقتراحات للبحث
            st.info("💡 **Search Tips:**\n"
                   "- Use more specific keywords\n"
                   "- Try synonyms of your search terms\n"
                   "- Check if the index is loaded correctly")
    
    # معلومات إضافية
    with st.expander("ℹ️ About this system"):
        st.markdown("""
        ### 📚 System Information
        
        **Dataset:** MS MARCO Passage
        - Testing: 5,000 documents
        - Full: 200,000 documents
        
        **Indexing:** Inverted Index with TF scoring
        
        **Preprocessing Pipeline:**
        1. Lowercasing
        2. Punctuation removal
        3. Number removal
        4. Stopword filtering
        5. Tokenization
        
        **Retrieval Models:**
        - **TF-IDF:** Term Frequency - Inverse Document Frequency
        - **BM25:** Okapi BM25 probabilistic model
        - **Hybrid Serial:** BM25 → BERT reranking
        - **Hybrid Parallel:** Multiple models with RRF fusion
        
        **Evaluation Metrics:** MAP, nDCG@10, P@10, Recall
        """)

if __name__ == "__main__":
    main()