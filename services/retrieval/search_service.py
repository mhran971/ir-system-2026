import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
import pandas as pd
from services.preprocessing.preprocessor import TextPreprocessor

class SearchService:
    def __init__(self):
        """تهيئة خدمة البحث"""
        self.index = None
        self.documents = {}
        self.preprocessor = TextPreprocessor()
        self.load_index()
        self.load_documents()
    
    def load_index(self):
        """تحميل الفهرس من القرص"""
        index_path = 'data/index/index.pkl'
        if os.path.exists(index_path):
            with open(index_path, 'rb') as f:
                self.index = pickle.load(f)
            print(f"✅ Index loaded: {len(self.index)} terms")
            
            # عرض مثال لهيكل الفهرس للتصحيح
            if self.index:
                sample_term = list(self.index.keys())[0]
                sample_value = self.index[sample_term]
                print(f"📊 Index structure sample: '{sample_term}' -> {type(sample_value)}")
                if isinstance(sample_value, list) and len(sample_value) > 0:
                    print(f"   Example entry: {sample_value[0]}")
        else:
            print(f"❌ Index not found at {index_path}")
            self.index = {}
    
    def load_documents(self):
        """تحميل الوثائق المعالجة مع نصها الأصلي"""
        # جرب تحميل ملفات مختلفة
        doc_files = [
            'data/processed/processed_docs_5000.pkl',
            'data/processed/processed_docs_200000.pkl',
            'data/raw/raw_docs.pkl'
        ]
        
        for doc_file in doc_files:
            if os.path.exists(doc_file):
                try:
                    with open(doc_file, 'rb') as f:
                        docs = pickle.load(f)
                    
                    # تخزين الوثائق في قاموس للوصول السريع
                    for doc in docs:
                        doc_id = str(doc['doc_id']) if isinstance(doc['doc_id'], (int, str)) else str(doc['doc_id'])
                        self.documents[doc_id] = {
                            'text': doc.get('original', doc.get('text', 'No text available')),
                            'processed_text': doc.get('processed_text', '')
                        }
                    
                    print(f"✅ Loaded {len(self.documents)} documents from {doc_file}")
                    return
                except Exception as e:
                    print(f"⚠️ Error loading {doc_file}: {e}")
        
        if not self.documents:
            print("⚠️ No documents loaded. Creating sample documents for testing...")
            self._create_sample_documents()
    
    def _create_sample_documents(self):
        """إنشاء وثائق تجريبية للاختبار"""
        sample_docs = {
            '2547': "Cloud storage is a model of computer data storage in which digital data is stored in logical pools across multiple servers.",
            '2567': "Data backup and recovery solutions for enterprise cloud storage systems provide redundancy.",
            '3520': "Cloud computing enables ubiquitous access to shared pools of configurable system resources.",
            '4723': "Storage area networks (SAN) are traditional alternatives to cloud storage.",
            '1685': "Public cloud storage services like AWS S3 offer scalable object storage."
        }
        
        for doc_id, text in sample_docs.items():
            self.documents[doc_id] = {
                'text': text,
                'processed_text': text.lower()
            }
        print(f"✅ Created {len(self.documents)} sample documents for testing")
    
    def get_document_text(self, doc_id):
        """الحصول على نص الوثيقة"""
        doc_id = str(doc_id)
        if doc_id in self.documents:
            return self.documents[doc_id]['text']
        return None
    
    def search(self, query, top_k=10):
        """البحث في الفهرس والعودة بالنتائج مع النص"""
        if not self.index:
            print("⚠️ No index loaded")
            return []
        
        # معالجة الاستعلام
        tokens = self.preprocessor.process(query)
        print(f"🔍 Query tokens: {tokens}")
        
        if not tokens:
            return []
        
        # حساب الدرجات
        scores = {}
        
        for token in tokens:
            if token in self.index:
                term_data = self.index[token]
                
                # معالجة مختلف أنواع هياكل البيانات
                if isinstance(term_data, dict):
                    # إذا كان القاموس: {doc_id: freq}
                    for doc_id, freq in term_data.items():
                        doc_id_str = str(doc_id)
                        scores[doc_id_str] = scores.get(doc_id_str, 0) + freq
                        
                elif isinstance(term_data, list):
                    # إذا كانت قائمة: [(doc_id, freq), ...]
                    for item in term_data:
                        if isinstance(item, (tuple, list)) and len(item) >= 2:
                            doc_id = str(item[0])
                            freq = item[1]
                            scores[doc_id] = scores.get(doc_id, 0) + freq
                        elif isinstance(item, dict):
                            # إذا كانت قواميس داخل القائمة
                            for doc_id, freq in item.items():
                                doc_id_str = str(doc_id)
                                scores[doc_id_str] = scores.get(doc_id_str, 0) + freq
                else:
                    # نوع غير متوقع
                    print(f"⚠️ Unexpected data type for token '{token}': {type(term_data)}")
        
        if not scores:
            print(f"⚠️ No matches found for tokens: {tokens}")
            return []
        
        # ترتيب النتائج
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # تنسيق النتائج مع النص
        results = []
        for doc_id, score in sorted_results:
            doc_text = self.get_document_text(doc_id)
            results.append({
                'doc_id': doc_id,
                'score': score,
                'text': doc_text[:500] + "..." if doc_text and len(doc_text) > 500 else doc_text,
                'full_text': doc_text
            })
        
        print(f"✅ Found {len(results)} results")
        return results

# اختبار سريع
if __name__ == "__main__":
    service = SearchService()
    
    print("\n" + "="*60)
    print("🔍 Search Service Test")
    print("="*60)
    
    test_queries = ['cloud storage', 'data backup', 'computing']
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        results = service.search(query)
        
        if results:
            for i, r in enumerate(results[:3], 1):
                print(f"   {i}. Doc {r['doc_id']} (Score: {r['score']})")
                if r['text']:
                    print(f"      {r['text'][:100]}...")
        else:
            print("   No results found.")
        print("-"*40)