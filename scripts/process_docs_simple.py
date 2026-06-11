import sys
import os
import pandas as pd
import pickle
import re
import string
from tqdm import tqdm

# قائمة stop words
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'were', 'will', 'with', 'i', 'you', 'we', 'they',
    'this', 'that', 'these', 'those', 'the', 'and', 'or', 'but', 'not'
}

def simple_preprocess(text):
    """وظيفة بسيطة لمعالجة النص"""
    if not text or not isinstance(text, str):
        return []
    
    # تحويل إلى حروف صغيرة
    text = text.lower()
    
    # إزالة الأرقام وعلامات الترقيم
    text = re.sub(r'[0-9]', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # تجزئة وتنظيف
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    
    return tokens

def main():
    print("=== Loading MS MARCO dataset ===")
    
    # تحميل البيانات مباشرة
    import ir_datasets
    dataset = ir_datasets.load('msmarco-passage/train')
    
    # تحديد عدد الوثائق
    max_docs = 10000  # ابدأ بـ 10,000 فقط للاختبار
    
    print(f"Loading {max_docs} documents...")
    docs = []
    for i, doc in enumerate(dataset.docs_iter()):
        if i >= max_docs:
            break
        docs.append({
            'doc_id': doc.doc_id,
            'text': doc.text
        })
        if (i + 1) % 1000 == 0:
            print(f"  Loaded {i + 1} documents")
    
    print(f"\n=== Preprocessing {len(docs)} documents ===")
    
    processed = []
    for doc in tqdm(docs, desc="Processing"):
        tokens = simple_preprocess(doc['text'])
        processed.append({
            'doc_id': doc['doc_id'],
            'tokens': tokens,
            'processed_text': ' '.join(tokens[:200])  # حد الطول
        })
    
    # حفظ النتائج
    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/processed_docs.pkl', 'wb') as f:
        pickle.dump(processed, f)
    
    print(f"\n✅ Done! Processed {len(processed)} documents")
    print(f"   Saved to data/processed/processed_docs.pkl")

if __name__ == "__main__":
    main()