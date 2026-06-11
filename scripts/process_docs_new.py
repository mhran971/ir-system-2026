# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
os.environ["PYTHONUTF8"] = "1"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ir_datasets
import pickle
import re
import string
from tqdm import tqdm

# قائمة stop words بسيطة
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'were', 'will', 'with', 'i', 'you', 'we', 'they',
    'this', 'that', 'these', 'those', 'the', 'and', 'or', 'but', 'not'
}

def preprocess_text(text):
    """دالة بسيطة لمعالجة النص"""
    if not text or not isinstance(text, str):
        return []
    
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    text = text.lower()
    text = re.sub(r'[0-9]', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    
    return tokens
def main():
    print("=" * 50)
    print("Processing MS MARCO Dataset")
    print("=" * 50)
    
    # عدد الوثائق (5000 للاختبار، 200000 للتشغيل الكامل)
    MAX_DOCS = 5000  # غير هذا إلى 200000 للتشغيل الكامل
    
    print(f"\n[1/4] Loading dataset...")
    print(f"Max documents: {MAX_DOCS}")
    
    # تحميل البيانات
    dataset = ir_datasets.load('msmarco-passage/train')
    
    # تحميل الوثائق
    documents = []
    for i, doc in enumerate(dataset.docs_iter()):
        if i >= MAX_DOCS:
            break
        documents.append({
            'doc_id': doc.doc_id,
            'text': doc.text
        })
        if (i + 1) % 1000 == 0:
            print(f"  Loaded {i + 1} documents")
    
    print(f"\n[2/4] Loaded {len(documents)} documents")
    
    # حفظ البيانات الخام
    os.makedirs('data/raw', exist_ok=True)
    with open('data/raw/raw_docs.pkl', 'wb') as f:
        pickle.dump(documents, f)
    print(f"  Saved raw docs to data/raw/raw_docs.pkl")
    
    # معالجة الوثائق
    print(f"\n[3/4] Preprocessing documents...")
    processed_docs = []
    
    for i, doc in enumerate(tqdm(documents, desc="Processing")):
        tokens = preprocess_text(doc['text'])
        processed_docs.append({
            'doc_id': doc['doc_id'],
            'tokens': tokens,
            'processed_text': ' '.join(tokens[:200])  # حفظ أول 200 كلمة فقط للتوفير
        })
        
        # عرض التقدم كل 500 وثيقة
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(documents)} docs")
    
    # حفظ البيانات المعالجة
    os.makedirs('data/processed', exist_ok=True)
    output_file = f'data/processed/processed_docs_{MAX_DOCS}.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(processed_docs, f)
    
    print(f"\n[4/4] Saved processed docs to {output_file}")
    
    # إحصائيات
    total_tokens = sum(len(doc['tokens']) for doc in processed_docs)
    avg_tokens = total_tokens / len(processed_docs) if processed_docs else 0
    
    print("\n" + "=" * 50)
    print("STATISTICS")
    print("=" * 50)
    print(f"Documents processed: {len(processed_docs)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Average tokens per doc: {avg_tokens:.2f}")
    print(f"\n✅ Success! File saved: {output_file}")
    
    # عرض مثال
    if processed_docs:
        print("\n📄 Example processed document:")
        example = processed_docs[0]
        print(f"  Doc ID: {example['doc_id']}")
        print(f"  Tokens: {example['tokens'][:20]}...")

if __name__ == "__main__":
    main()