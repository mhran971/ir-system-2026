# scripts/process_docs.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ir_datasets
import pickle
import re
import string
from tqdm import tqdm

# قائمة stop words
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'were', 'will', 'with', 'i', 'you', 'we', 'they',
    'this', 'that', 'these', 'those'
}

def preprocess_text(text):
    """
    معالجة النص: تطبيع، تنظيف، تجزئة
    """
    if not text or not isinstance(text, str):
        return []
    
    # تحويل إلى حروف صغيرة
    text = text.lower()
    
    # إزالة الأرقام
    text = re.sub(r'[0-9]', ' ', text)
    
    # إزالة علامات الترقيم
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # تجزئة وتنظيف
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    
    return tokens

def main():
    print("=" * 60)
    print("📚 MS MARCO Dataset Processor")
    print("=" * 60)
    
    # عدد الوثائق (5000 للاختبار، 200000 للتشغيل الكامل)
    MAX_DOCS = 5000  # غير هذا الرقم إلى 200000 للبيانات الكاملة
    
    print(f"\n📊 Configuration:")
    print(f"   Dataset: msmarco-passage/train")
    print(f"   Max documents: {MAX_DOCS}")
    
    # تحميل البيانات
    print(f"\n[1/4] Loading dataset...")
    dataset = ir_datasets.load('msmarco-passage/train')
    
    documents = []
    for i, doc in enumerate(dataset.docs_iter()):
        if i >= MAX_DOCS:
            break
        documents.append({
            'doc_id': str(doc.doc_id),
            'text': doc.text,
            'original': doc.text
        })
        if (i + 1) % 1000 == 0:
            print(f"   Loaded {i + 1}/{MAX_DOCS} documents")
    
    print(f"\n[2/4] Loaded {len(documents)} documents")
    
    # حفظ البيانات الخام
    os.makedirs('data/raw', exist_ok=True)
    with open('data/raw/raw_docs.pkl', 'wb') as f:
        pickle.dump(documents, f)
    print(f"   ✅ Saved raw documents to data/raw/raw_docs.pkl")
    
    # معالجة الوثائق
    print(f"\n[3/4] Preprocessing documents...")
    processed_docs = []
    
    for doc in tqdm(documents, desc="Processing"):
        tokens = preprocess_text(doc['text'])
        processed_docs.append({
            'doc_id': doc['doc_id'],
            'original': doc['text'],  # النص الأصلي
            'text': doc['text'],      # النص الأصلي
            'tokens': tokens,
            'processed_text': ' '.join(tokens)
        })
    
    # حفظ البيانات المعالجة
    os.makedirs('data/processed', exist_ok=True)
    output_file = f'data/processed/processed_docs_{MAX_DOCS}.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(processed_docs, f)
    
    # حفظ نسخة باسم عام أيضاً
    with open('data/processed/processed_docs.pkl', 'wb') as f:
        pickle.dump(processed_docs, f)
    
    print(f"\n[4/4] ✅ Saved processed documents to {output_file}")
    
    # إحصائيات
    total_tokens = sum(len(doc['tokens']) for doc in processed_docs)
    avg_tokens = total_tokens / len(processed_docs) if processed_docs else 0
    
    print("\n" + "=" * 60)
    print("📊 STATISTICS")
    print("=" * 60)
    print(f"   Documents processed: {len(processed_docs)}")
    print(f"   Total tokens: {total_tokens:,}")
    print(f"   Average tokens per doc: {avg_tokens:.2f}")
    
    # عرض مثال
    if processed_docs:
        print("\n📄 Example processed document:")
        example = processed_docs[0]
        print(f"   Doc ID: {example['doc_id']}")
        print(f"   Original text: {example['original'][:100]}...")
        print(f"   Tokens: {example['tokens'][:15]}...")
    
    print("\n✅ Done! You can now run: python scripts/build_index.py")

if __name__ == "__main__":
    main()