# services/retrieval/search_service.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
import pandas as pd
from services.preprocessing.preprocessor import TextPreprocessor
from services.query_processing.query_processor import QueryProcessor

class SearchService:
    """
    خدمة البحث - تقوم بالبحث في الفهرس وإرجاع النتائج مع النص الكامل
    """
    
    def __init__(self, query_processor=None):
        self.index = None
        self.documents = {}  # تخزين محتوى الوثائق
        self.query_processor = query_processor or QueryProcessor()
        # For backward compatibility if other modules access self.preprocessor
        self.preprocessor = self.query_processor.preprocessor
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
        """تحميل الوثائق الكاملة مع نصها الأصلي من ملفات MS MARCO"""
        
        # قائمة بجميع ملفات الوثائق المحتملة (حسب العدد)
        doc_files = [
            ('data/processed/processed_docs_200000.pkl', 200000),
            ('data/processed/processed_docs_5000.pkl', 5000),
            ('data/processed/processed_docs_10000.pkl', 10000),
            ('data/processed/processed_docs.pkl', None),
            ('data/raw/raw_docs.pkl', None)
        ]
        
        for doc_file, expected_count in doc_files:
            if os.path.exists(doc_file):
                try:
                    with open(doc_file, 'rb') as f:
                        docs = pickle.load(f)
                    
                    print(f"📂 Loading documents from {doc_file}")
                    
                    # تخزين الوثائق في قاموس للوصول السريع
                    loaded_count = 0
                    for doc in docs:
                        doc_id = str(doc.get('doc_id', ''))
                        if doc_id:
                            # حفظ النص الأصلي (الأولوية: original ثم text)
                            text = doc.get('original', doc.get('text', ''))
                            if text and len(text) > 10:  # تأكد أن النص غير فارغ
                                self.documents[doc_id] = {
                                    'text': text,
                                    'processed_text': doc.get('processed_text', ''),
                                    'tokens': doc.get('tokens', [])
                                }
                                loaded_count += 1
                    
                    if self.documents:
                        print(f"✅ Loaded {len(self.documents)} documents from {doc_file}")
                        return
                    else:
                        print(f"⚠️ No valid documents found in {doc_file}")
                        
                except Exception as e:
                    print(f"⚠️ Error loading {doc_file}: {e}")
        
        # إذا لم يتم العثور على وثائق، حاول تحميل CSV
        csv_file = 'data/raw/raw_docs.csv'
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    doc_id = str(row.get('doc_id', ''))
                    text = row.get('text', '')
                    if doc_id and text:
                        self.documents[doc_id] = {
                            'text': text,
                            'processed_text': '',
                            'tokens': []
                        }
                print(f"✅ Loaded {len(self.documents)} documents from CSV")
                return
            except Exception as e:
                print(f"⚠️ Error loading CSV: {e}")
        
        # إذا لم يتم العثور على أي ملفات،提示 المستخدم بتشغيل المعالجة
        if not self.documents:
            print("\n" + "="*60)
            print("⚠️ NO DOCUMENTS FOUND!")
            print("="*60)
            print("Please run the following commands first:")
            print("  1. python scripts/process_docs.py")
            print("  2. python scripts/build_index.py")
            print("\nOr to process the full dataset (200,000 documents):")
            print("  - Edit scripts/process_docs.py and set MAX_DOCS = 200000")
            print("  - Then run: python scripts/process_docs.py")
            print("  - Then run: python scripts/build_index.py")
            print("="*60)
            
            # لا نقوم بإنشاء عينات تجريبية، بل نرفع خطأ
            raise Exception("No documents found. Please run process_docs.py first!")
    
    def get_document_text(self, doc_id):
        """الحصول على نص الوثيقة بالكامل"""
        doc_id = str(doc_id)
        if doc_id in self.documents:
            return self.documents[doc_id]['text']
        return None
    
    def get_document_info(self, doc_id):
        """الحصول على معلومات كاملة عن الوثيقة"""
        doc_id = str(doc_id)
        if doc_id in self.documents:
            return self.documents[doc_id]
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
            
            # إذا لم يتم العثور على النص، ابحث في الوثائق المحملة
            if not doc_text and doc_id in self.documents:
                doc_text = self.documents[doc_id].get('text', 'No text available')
            
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
        print(f"📄 Documents loaded: {len(service.documents)}")
        
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