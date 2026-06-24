# scripts/process_docs.py
"""
Data loading and preprocessing script.

Run ONCE to prepare documents for indexing:
    python scripts/process_docs.py

What it does:
    1. Loads raw documents from ClinicalTrials via ir_datasets
    2. Calls TextPreprocessor service to clean and tokenize
    3. Saves processed documents to data/processed/

This script is just an ORCHESTRATOR.
All actual logic lives in services/preprocessing/.
"""

import sys
import os

# ── UTF-8 fix for Windows — must be before any other import ──────────────────
sys.stdout.reconfigure(encoding='utf-8')
os.environ["PYTHONUTF8"] = "1"

# ── Patch ir_datasets to use UTF-8 (same patch as loader.py) ─────────────────
import ir_datasets.formats.tsv as _tsv
import io as _io
_orig_wrapper = _tsv.io.TextIOWrapper
def _utf8_wrapper(buffer, *args, **kwargs):
    kwargs.setdefault("encoding", "utf-8")
    kwargs.setdefault("errors", "replace")
    return _orig_wrapper(buffer, *args, **kwargs)
_tsv.io.TextIOWrapper = _utf8_wrapper

# ── Now safe to import everything else ───────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import ir_datasets
from tqdm import tqdm

# ── Import the preprocessing SERVICE ─────────────────────────────────────────
from services.preprocessing.preprocessor import TextPreprocessor


def main():
    print("=" * 60)
    print("📚 ClinicalTrials Dataset Processor")
    print("=" * 60)

    print(f"\n📊 Configuration:")
    print(f"   Dataset  : clinicaltrials/2017/trec-pm-2017")

    # ── Step 1: Load raw documents ────────────────────────────────────────────
    print(f"\n[1/3] Loading dataset...")
    dataset = ir_datasets.load('clinicaltrials/2017/trec-pm-2017')

    raw_docs = []
    for i, doc in enumerate(dataset.docs_iter()):
        fields = [doc.title, doc.condition, doc.summary, doc.detailed_description, doc.eligibility]
        merged_text = " ".join([f for f in fields if f])
        raw_docs.append({
            'doc_id':   str(doc.doc_id),
            'text':     merged_text,       # merged clinical trial fields
            'original': merged_text,       # kept as alias for compatibility
        })
        if (i + 1) % 10000 == 0:
            print(f"   Loaded {i + 1} documents")

    print(f"   ✅ Loaded {len(raw_docs)} documents")

    # Save raw docs
    os.makedirs('data/raw', exist_ok=True)
    with open('data/raw/raw_docs.pkl', 'wb') as f:
        pickle.dump(raw_docs, f)
    print(f"   ✅ Saved raw docs → data/raw/raw_docs.pkl")

    # ── Step 2: Preprocess using the TextPreprocessor SERVICE ─────────────────
    print(f"\n[2/3] Preprocessing with TextPreprocessor service...")
    preprocessor = TextPreprocessor(use_stemming=False, use_lemmatization=True)

    processed_docs = []
    for doc in tqdm(raw_docs, desc="Preprocessing"):
        tokens = preprocessor.process(doc['text'])   # ← calls the SERVICE
        processed_docs.append({
            'doc_id':         doc['doc_id'],
            'text':           doc['text'],            # original text (for BERT + UI display)
            'original':       doc['text'],            # alias
            'tokens':         tokens,                 # for inverted index + VSM
            'processed_text': ' '.join(tokens),       # joined string version
        })

    # ── Step 3: Save processed documents ─────────────────────────────────────
    print(f"\n[3/3] Saving processed documents...")
    os.makedirs('data/processed', exist_ok=True)

    # Save to data/processed/processed_docs.pkl
    processed_path = 'data/processed/processed_docs.pkl'
    with open(processed_path, 'wb') as f:
        pickle.dump(processed_docs, f)
    print(f"   ✅ Saved → {processed_path}")

    # Save queries and qrels
    queries = {q.query_id: (q.text if hasattr(q, 'text') else q.default_text()) for q in dataset.queries_iter()}
    queries_path = 'data/processed/queries.pkl'
    with open(queries_path, 'wb') as f:
        pickle.dump(queries, f)
    print(f"   ✅ Saved queries → {queries_path}")

    qrels = [(q.query_id, q.doc_id, q.relevance) for q in dataset.qrels_iter()]
    qrels_path = 'data/processed/qrels.pkl'
    with open(qrels_path, 'wb') as f:
        pickle.dump(qrels, f)
    print(f"   ✅ Saved qrels → {qrels_path}")

    # Save the preprocessor to ensure query consistency
    preprocessor_path = 'data/processed/preprocessor.pkl'
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"   ✅ Saved preprocessor → {preprocessor_path}")

    # ── Statistics ────────────────────────────────────────────────────────────
    total_tokens = sum(len(d['tokens']) for d in processed_docs)
    avg_tokens   = total_tokens / len(processed_docs) if processed_docs else 0

    print("\n" + "=" * 60)
    print("📊 STATISTICS")
    print("=" * 60)
    print(f"   Documents processed  : {len(processed_docs)}")
    print(f"   Total tokens         : {total_tokens:,}")
    print(f"   Avg tokens / doc     : {avg_tokens:.1f}")

    if processed_docs:
        ex = processed_docs[0]
        print(f"\n📄 Example doc [{ex['doc_id']}]:")
        print(f"   text   : {ex['text'][:100]}...")
        print(f"   tokens : {ex['tokens'][:10]}...")

    print(f"\n✅ Done! Next step:")
    print(f"   python scripts/build_index.py")


if __name__ == "__main__":
    main()
