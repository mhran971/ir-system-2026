# services/retrieval/search_service.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
from services.preprocessing.preprocessor import TextPreprocessor

class SearchService:
    """
    خدمة البحث - تقوم بالبحث في الفهرس وإرجاع النتائج مع النص الكامل
    """
    
    def __init__(self):
        self.index = None
        self.documents = {}  # تخزين محتوى الوثائق
        self.preprocessor = TextPreprocessor()
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
                    print(f"✅ Index loaded: {len(self.index)} terms")
                    return
                except Exception as e:
                    print(f"⚠️ Error loading index: {e}")
        
        print("❌ No index found. Run scripts/build_index.py first")
        self.index = {}
    
    def load_documents(self):
        """تحميل الوثائق مع نصها الأصلي"""
        doc_files = [
            'data/processed/processed_docs_5000.pkl',
            'data/processed/processed_docs.pkl',
            'data/raw/raw_docs.pkl'
        ]
        
        for doc_file in doc_files:
            if os.path.exists(doc_file):
                try:
                    with open(doc_file, 'rb') as f:
                        docs = pickle.load(f)
                    
                    for doc in docs:
                        doc_id = str(doc.get('doc_id', ''))
                        if doc_id:
                            # حفظ النص الأصلي
                            text = doc.get('original', doc.get('text', ''))
                            if text:
                                self.documents[doc_id] = {
                                    'text': text,
                                    'processed_text': doc.get('processed_text', '')
                                }
                    
                    if self.documents:
                        print(f"✅ Loaded {len(self.documents)} documents from {doc_file}")
                        return
                except Exception as e:
                    print(f"⚠️ Error loading {doc_file}: {e}")
        
        # إذا لم يتم العثور على وثائق، أنشئ عينات للاختبار
        if not self.documents:
            print("⚠️ No documents found. Creating sample documents...")
            self._create_sample_documents()
    
    def _create_sample_documents(self):
        """إنشاء وثائق تجريبية للاختبار"""
        sample_docs = {
            '2547': "Cloud storage is a model of computer data storage in which digital data is stored in logical pools across multiple servers. This enables organizations to store, manage, and access data over the internet. Major providers include AWS S3, Google Cloud Storage, and Microsoft Azure Blob Storage.",
            
            '2567': "Data backup and recovery solutions for enterprise cloud storage systems provide redundancy and disaster recovery capabilities. Regular backups ensure data can be restored in case of hardware failure or cyber attacks.",
            
            '3520': "Cloud computing enables ubiquitous access to shared pools of configurable system resources and higher-level services. It provides on-demand network access to computing resources.",
            
            '4723': "Storage area networks (SAN) and network-attached storage (NAS) are traditional alternatives to cloud storage solutions for on-premises infrastructure. These solutions require physical hardware maintenance.",
            
            '1685': "Public cloud storage services like AWS S3, Google Cloud Storage, and Azure Blob Storage offer scalable object storage with pay-as-you-go pricing models. Users can store and retrieve any amount of data from anywhere."
        }
        
        for doc_id, text in sample_docs.items():
            self.documents[doc_id] = {
                'text': text,
                'processed_text': text.lower()
            }
        print(f"✅ Created {len(self.documents)} sample documents for testing")
    
    def get_document_text(self, doc_id):
        """الحصول على نص الوثيقة بالكامل"""
        doc_id = str(doc_id)
        if doc_id in self.documents:
            return self.documents[doc_id]['text']
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
        tokens = self.preprocessor.process(query)
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
                        scores[str(doc_id)] = scores.get(str(doc_id), 0) + freq
                        
                elif isinstance(term_data, list):
                    for item in term_data:
                        if isinstance(item, (tuple, list)) and len(item) >= 2:
                            scores[str(item[0])] = scores.get(str(item[0]), 0) + item[1]
                        elif isinstance(item, dict):
                            for doc_id, freq in item.items():
                                scores[str(doc_id)] = scores.get(str(doc_id), 0) + freq
        
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
        
        print(f"✅ Found {len(results)} results")
        return results

# اختبار سريع
if __name__ == "__main__":
    service = SearchService()
    
    print("\n" + "="*60)
    print("🔍 Search Service Test")
    print("="*60)
    
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
                    print(f"   📄 {r['text'][:200]}...")
                else:
                    print(f"   ⚠️ No text available")
        else:
            print("   No results found.")
        print("-"*40)