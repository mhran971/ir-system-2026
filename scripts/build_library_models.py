# scripts/build_library_models.py
import os
import sys
import time
import pickle
import shutil
import scipy.sparse
from pathlib import Path

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from services.retrieval.vsm_search_service import identity_analyzer


def get_index_dir() -> str:
    default_dir = 'data/index'
    try:
        abs_path = os.path.abspath(default_dir)
        drive = os.path.splitdrive(abs_path)[0]
        usage = shutil.disk_usage(drive)
        if usage.free < 1.5 * 1024 * 1024 * 1024:
            fallback_dir = os.path.expanduser('~/.ir_system_cache')
            os.makedirs(fallback_dir, exist_ok=True)
            print(f"ℹ️ [BuildModels] Target drive space is low. Using C: drive cache: {fallback_dir}")
            return fallback_dir
    except Exception as e:
        print(f"⚠️ [BuildModels] Disk check failed: {e}")
    os.makedirs(default_dir, exist_ok=True)
    return default_dir


def main():
    print("=" * 60)
    print("🚀 Fitting and Serializing Library-based Search Models")
    print("=" * 60)

    processed_path = 'data/processed/processed_docs.pkl'
    if not os.path.exists(processed_path):
        print(f"❌ Error: Processed documents file not found at {processed_path}")
        return

    print("⏳ Loading processed documents...")
    t0 = time.time()
    with open(processed_path, 'rb') as f:
        docs = pickle.load(f)
    print(f"✅ Loaded {len(docs):,} documents in {time.time() - t0:.2f}s")

    doc_ids = []
    corpus_tokens = []
    corpus_texts = []
    
    for doc in docs:
        doc_id = str(doc.get('doc_id', ''))
        if doc_id:
            doc_ids.append(doc_id)
            tokens = doc.get('tokens', [])
            corpus_tokens.append(tokens)
            corpus_texts.append(' '.join(tokens) if tokens else '')

    index_dir = get_index_dir()

    # ===== 1. Fit and Save VSM Scikit-Learn Model =====
    print("\n--- 1. Fitting scikit-learn TfidfVectorizer ---")
    t0 = time.time()
    
    vectorizer = TfidfVectorizer(
        analyzer=identity_analyzer,
        lowercase=False,
        token_pattern=None
    )
    
    print("⏳ Fitting TfidfVectorizer and transforming corpus...")
    tfidf_matrix = vectorizer.fit_transform(corpus_tokens)
    print(f"✅ VSM fit completed in {time.time() - t0:.2f}s")
    print(f"   Matrix shape: {tfidf_matrix.shape}")
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_):,}")

    # حفظ نموذج VSM
    vsm_model_path = os.path.join(index_dir, 'vsm_model.pkl')
    vsm_matrix_path = os.path.join(index_dir, 'vsm_matrix.npz')
    
    print(f"⏳ Saving VSM model to {vsm_model_path}...")
    vsm_data = {
        'vectorizer': vectorizer,
        'doc_ids': doc_ids
    }
    with open(vsm_model_path, 'wb') as f:
        pickle.dump(vsm_data, f)
        
    print(f"⏳ Saving TF-IDF sparse matrix to {vsm_matrix_path}...")
    scipy.sparse.save_npz(vsm_matrix_path, tfidf_matrix)
    print("✅ VSM models successfully saved!")

    # ===== 2. Fit and Save rank_bm25 Model =====
    print("\n--- 2. Fitting rank_bm25 BM25Okapi ---")
    t0 = time.time()
    
    print("⏳ Initializing BM25Okapi on corpus...")
    bm25 = BM25Okapi(corpus_tokens, k1=1.5, b=0.75)
    print(f"✅ BM25 fit completed in {time.time() - t0:.2f}s")

    # حفظ نموذج BM25
    bm25_model_path = os.path.join(index_dir, 'bm25_model.pkl')
    print(f"⏳ Saving BM25 model to {bm25_model_path}...")
    bm25_data = {
        'bm25': bm25,
        'doc_ids': doc_ids
    }
    with open(bm25_model_path, 'wb') as f:
        pickle.dump(bm25_data, f)
    print("✅ BM25 model successfully saved!")

    # ===== 3. إحصائيات =====
    print("\n" + "=" * 60)
    print("📊 MODEL STATISTICS")
    print("=" * 60)
    print(f"   Total documents: {len(doc_ids):,}")
    print(f"   Average document length: {bm25.avgdl:.2f}")
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_):,}")
    print(f"   TF-IDF matrix: {tfidf_matrix.shape[0]:,} × {tfidf_matrix.shape[1]:,}")
    
    print("\n" + "=" * 60)
    print("🎉 All library-based models have been built and saved successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()