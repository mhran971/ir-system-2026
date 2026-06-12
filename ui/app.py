# ui/app.py
import streamlit as st
import sys
import os
import time

# إضافة المسار
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.retrieval.search_service import SearchService

# تهيئة خدمة البحث (مع caching لتحسين الأداء)
@st.cache_resource
def get_search_service():
    return SearchService()

def main():
    st.set_page_config(
        page_title="IR System 2026",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Information Retrieval System 2026")
    st.markdown("---")
    
    # شريط جانبي للإعدادات
    with st.sidebar:
        st.header("⚙️ Settings")
        
        model = st.selectbox(
            "📊 Retrieval Model",
            ["Simple TF-IDF", "BM25", "Hybrid Serial", "Hybrid Parallel"],
            help="Select the ranking algorithm"
        )
        
        top_k = st.slider("📄 Number of results", 5, 50, 10)
        
        st.markdown("---")
        
        # خيارات العرض
        st.subheader("🎨 Display Options")
        show_text = st.checkbox("📖 Show document text", value=True)
        max_text_length = st.slider("📏 Max text length", 100, 1000, 300)
        show_matching = st.checkbox("🔍 Show matching terms", value=True)
        
        st.markdown("---")
        
        # إحصائيات
        try:
            service = get_search_service()
            st.info(f"📊 **System Stats**\n\n"
                   f"🔤 Terms in index: {len(service.index):,}\n"
                   f"📄 Documents: {len(service.documents):,}\n"
                   f"🎯 Active model: {model}")
        except Exception as e:
            st.error(f"Error loading service: {e}")
    
    # منطقة البحث الرئيسية
    st.header("📝 Enter your search query")
    
    # أمثلة للاستعلامات
    example_queries = ["cloud storage backup", "machine learning", "data recovery", "artificial intelligence"]
    cols = st.columns(len(example_queries))
    for i, example in enumerate(example_queries):
        with cols[i]:
            if st.button(f"🔍 {example}", key=f"btn_{i}"):
                st.session_state.query = example
    
    # مربع إدخال الاستعلام
    query = st.text_input(
        "Search query:",
        value=st.session_state.get("query", ""),
        placeholder="e.g., cloud storage backup solutions for enterprise...",
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # نتائج البحث
    if search_button and query:
        start_time = time.time()
        
        with st.spinner("🔍 Searching..."):
            try:
                service = get_search_service()
                results = service.search(query, top_k=top_k)
            except Exception as e:
                st.error(f"Error during search: {e}")
                results = []
        
        search_time = time.time() - start_time
        
        if results:
            st.success(f"✅ Found {len(results)} results in {search_time:.3f} seconds")
            st.markdown("---")
            
            # عرض النتائج
            for i, result in enumerate(results, 1):
                # عنوان النتيجة
                st.markdown(f"### {i}. 📄 Document `{result['doc_id']}`")
                st.markdown(f"**🎯 Relevance Score:** `{result['score']:.2f}`")
                
                # عرض النص إذا كان متاحاً ومطلوباً
                if show_text and result.get('text'):
                    st.markdown("**📖 Document Content:**")
                    text_preview = result['text'][:max_text_length]
                    if len(result['text']) > max_text_length:
                        text_preview += "..."
                    
                    # عرض النص في صندوق جميل
                    st.markdown(f"> {text_preview}")
                    
                    # زر لتوسيع النص
                    with st.expander("📚 Show full document"):
                        st.write(result.get('full_text', result['text']))
                
                # عرض الكلمات المفتاحية التي ظهرت
                if show_matching:
                    with st.expander("🔍 Show matching terms"):
                        try:
                            tokens = service.preprocessor.process(query)
                            if tokens:
                                st.write("**Query terms found in this document:**")
                                doc_matches = []
                                for token in tokens:
                                    if token in service.index:
                                        term_data = service.index[token]
                                        doc_id = result['doc_id']
                                        
                                        # البحث في أنواع مختلفة من هياكل الفهرس
                                        found = False
                                        freq = 0
                                        
                                        if isinstance(term_data, dict) and doc_id in term_data:
                                            freq = term_data[doc_id]
                                            found = True
                                        elif isinstance(term_data, list):
                                            for item in term_data:
                                                if isinstance(item, (tuple, list)) and len(item) >= 2 and str(item[0]) == doc_id:
                                                    freq = item[1]
                                                    found = True
                                                    break
                                                elif isinstance(item, dict) and doc_id in item:
                                                    freq = item[doc_id]
                                                    found = True
                                                    break
                                        
                                        if found:
                                            doc_matches.append(f"- `{token}`: appears **{freq}** time(s)")
                                        else:
                                            doc_matches.append(f"- `{token}`: not found in this document")
                                
                                if doc_matches:
                                    for match in doc_matches:
                                        st.write(match)
                                else:
                                    st.write("No term matches found")
                            else:
                                st.write("No query terms after preprocessing")
                        except Exception as e:
                            st.write(f"Error showing matching terms: {e}")
                
                st.markdown("---")
        else:
            st.warning("⚠️ No results found. Try different keywords.")
            
            # اقتراحات للبحث
            st.info("💡 **Search Tips:**\n"
                   "- Try using more specific keywords\n"
                   "- Use synonyms of your search terms\n"
                   "- Make sure the index is loaded correctly\n"
                   "- Try running `python scripts/build_index.py` first")
    
    # معلومات إضافية في الأسفل
    with st.expander("ℹ️ About this system"):
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
        
        **Indexing:** Inverted Index with TF scoring
        
        **Retrieval Models:**
        - **TF-IDF:** Term Frequency - Inverse Document Frequency
        - **BM25:** Okapi BM25 probabilistic model
        - **Hybrid Serial:** BM25 → BERT reranking
        - **Hybrid Parallel:** Multiple models with RRF fusion
        
        **Evaluation Metrics:** MAP, nDCG@10, P@10, Recall
        
        **Technologies:** Python, Streamlit, FastAPI, scikit-learn, NLTK
        """)

if __name__ == "__main__":
    main()