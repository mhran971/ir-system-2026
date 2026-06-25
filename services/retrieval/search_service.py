# services/retrieval/search_service.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
import pandas as pd
from services.preprocessing.preprocessor import TextPreprocessor
from services.query_processing.query_processor import QueryProcessor
from services.indexing.document_store import DocumentStore

class SearchService:
    """
    خدمة البحث - تقوم بالبحث في الفهرس وإرجاع النتائج مع النص الكامل
    """
    
    def __init__(self, query_processor=None):
        self.index = None
        self.query_processor = query_processor or QueryProcessor()
        # For backward compatibility if other modules access self.preprocessor
        self.preprocessor = self.query_processor.preprocessor
        self.document_store = DocumentStore()
        
        self.load_index()
        self.load_documents()
    
    def load_index(self):
        """تحميل الفهرس من القرص"""
        index_paths = [
            'data/index/index.pkl',
            'data/index/inverted_index.pkl'
        ]
        
        for index_path in index_paths:
            if os.path.exists(index_path):
                try:
                    with open(index_path, 'rb') as f:
                        self.index = pickle.load(f)
                    print(f"✅ Index loaded: {len(self.index)} terms from {index_path}")
                    return
                except Exception as e:
                    print(f"⚠️ Error loading index: {e}")
        
        print("❌ No index found. Run scripts/build_index.py first")
        self.index = {}
    
    def load_documents(self):
        """تهيئة مخزن المستندات الخفيف (SQLite) بدلاً من تحميل pickle الضخم بالكامل في الذاكرة"""
        doc_files = [
            'data/processed/processed_docs.pkl',
        ]
        # تهيئة مخزن المستندات (الذي يقوم بالتحميل السريع من SQLite أو الترحيل التلقائي)
        self.document_store.load(doc_files)
        print(f"✅ Document store initialized with {self.document_store.total_docs} documents (SQLite-backed)")
    
    def get_document_text(self, doc_id):
        """الحصول على نص الوثيقة بالكامل"""
        doc_id = str(doc_id)
        doc = self.document_store.get_doc(doc_id)
        return doc['text'] if doc else None
    
    def get_document_info(self, doc_id):
        """الحصول على معلومات كاملة عن الوثيقة"""
        doc_id = str(doc_id)
        doc = self.document_store.get_doc(doc_id)
        if doc:
            return {
                'text': doc['text'],
                'processed_text': '',
                'tokens': []
            }
        return None
    
    def search(self, query, top_k=10):
        """
        البحث في الفهرس وإرجاع النتائج المرتبة
        
        Args:
            query: نص الاستعلام
            top_k: عدد النتائج المطلوبة
        
        Returns:
            list: قائمة بالنتائج كل نتيجة تحتوي على doc_id, score, text
        """
        if not self.index:
            print("⚠️ No index loaded")
            return []
        
        # معالجة الاستعلام
        tokens = self.query_processor.process_query(query)
        print(f"🔍 Query: '{query}' -> Tokens: {tokens}")
        
        if not tokens:
            return []
        
        # حساب الدرجات لكل وثيقة
        scores = {}
        
        for token in tokens:
            if token in self.index:
                term_data = self.index[token]
                
                # معالجة أنواع مختلفة من هياكل الفهرس
                if isinstance(term_data, dict):
                    for doc_id, freq in term_data.items():
                        doc_id_str = str(doc_id)
                        scores[doc_id_str] = scores.get(doc_id_str, 0) + freq
                        
                elif isinstance(term_data, list):
                    for item in term_data:
                        if isinstance(item, (tuple, list)) and len(item) >= 2:
                            doc_id_str = str(item[0])
                            freq = item[1]
                            scores[doc_id_str] = scores.get(doc_id_str, 0) + freq
                        elif isinstance(item, dict):
                            for doc_id, freq in item.items():
                                doc_id_str = str(doc_id)
                                scores[doc_id_str] = scores.get(doc_id_str, 0) + freq
                else:
                    # نوع غير متوقع
                    print(f"⚠️ Unexpected data type for token '{token}': {type(term_data)}")
        
        if not scores:
            print(f"⚠️ No matches found for tokens: {tokens}")
            return []
        
        # ترتيب النتائج تنازلياً حسب الدرجة
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # إضافة النص الكامل للنتائج
        results = []
        for doc_id, score in sorted_results:
            doc_text = self.get_document_text(doc_id)
            
            results.append({
                'doc_id': doc_id,
                'score': score,
                'text': doc_text if doc_text else "No text available",
                'full_text': doc_text if doc_text else "No text available"
            })
            
            # للتصحيح: طباعة إذا تم العثور على النص
            if doc_text and doc_text != "No text available":
                print(f"   ✅ Found text for doc {doc_id}: {doc_text[:50]}...")
            else:
                print(f"   ⚠️ No text found for doc {doc_id}")
        
        print(f"✅ Found {len(results)} results")
        return results

# اختبار سريع
if __name__ == "__main__":
    try:
        service = SearchService()
        
        print("\n" + "="*60)
        print("🔍 Search Service Test (Full Dataset)")
        print("="*60)
        print(f"📊 Index size: {len(service.index)} terms")
        print(f"📄 Documents loaded: {service.document_store.total_docs}")
        
        while True:
            query = input("\nEnter search query (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
            
            results = service.search(query)
            
            if results:
                print(f"\n📊 Top {len(results)} results:")
                for i, r in enumerate(results[:5], 1):
                    print(f"\n{i}. Document {r['doc_id']} (Score: {r['score']})")
                    if r['text'] and r['text'] != "No text available":
                        # عرض أول 150 حرف من النص
                        preview = r['text'][:150].replace('\n', ' ')
                        print(f"   📄 {preview}...")
                    else:
                        print(f"   ⚠️ No text available - document may not be loaded")
            else:
                print("   No results found.")
            print("-"*40)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure you have run:")
        print("  1. python scripts/process_docs.py")
        print("  2. python scripts/build_index.py")