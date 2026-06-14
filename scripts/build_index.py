# scripts/build_index.py
"""
Inverted index builder script.

Run ONCE after process_docs.py:
    python scripts/build_index.py

What it does:
    1. Loads processed documents from data/processed/
    2. Calls InvertedIndex service to build the index
    3. Saves the index to data/index/

This script is just an ORCHESTRATOR.
All actual logic lives in services/indexing/inverted_index.py.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle

# ── Import the indexing SERVICE ───────────────────────────────────────────────
from services.indexing.inverted_index import InvertedIndex


# Where to look for processed docs (priority order)
PROCESSED_DOC_PATHS = [
    'data/processed/processed_docs_200000.pkl',
    'data/processed/processed_docs_5000.pkl',
    'data/processed/processed_docs.pkl',
]

INDEX_SAVE_PATH = 'data/index/inverted_index.pkl'


def load_processed_docs() -> list:
    """Find and load the processed documents file."""
    for path in PROCESSED_DOC_PATHS:
        if os.path.exists(path):
            print(f"📂 Loading: {path}")
            with open(path, 'rb') as f:
                docs = pickle.load(f)
            print(f"   ✅ Loaded {len(docs)} documents")
            return docs

    print("❌ No processed documents found.")
    print("   Run: python scripts/process_docs.py  first.")
    return []


def main():
    print("=" * 60)
    print("🔨 Building Inverted Index")
    print("=" * 60)

    # ── Step 1: Load processed documents ─────────────────────────────────────
    print("\n[1/3] Loading processed documents...")
    docs = load_processed_docs()
    if not docs:
        return

    # ── Step 2: Build index using the InvertedIndex SERVICE ──────────────────
    print("\n[2/3] Building inverted index...")
    index = InvertedIndex()          # ← calls the SERVICE

    for i, doc in enumerate(docs):
        doc_id = str(doc['doc_id'])
        tokens = doc.get('tokens', [])
        index.add_document(doc_id, tokens)   # ← SERVICE method

        if (i + 1) % 1000 == 0:
            print(f"   Indexed {i + 1}/{len(docs)} documents")

    stats = index.get_stats()
    print(f"\n   ✅ Index built:")
    print(f"      Unique terms     : {stats['unique_terms']:,}")
    print(f"      Total documents  : {stats['total_documents']:,}")
    print(f"      Total entries    : {stats['total_entries']:,}")

    # ── Step 3: Save index to disk ────────────────────────────────────────────
    print(f"\n[3/3] Saving index...")
    os.makedirs('data/index', exist_ok=True)
    with open(INDEX_SAVE_PATH, 'wb') as f:
        pickle.dump(index, f)

    print(f"   ✅ Saved → {INDEX_SAVE_PATH}")

    print(f"\n✅ Done! Next steps:")
    print(f"   python scripts/build_bert_index.py   (for BERT embeddings)")
    print(f"   streamlit run ui/app.py               (to launch the UI)")


if __name__ == "__main__":
    main()