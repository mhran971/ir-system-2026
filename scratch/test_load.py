# scratch/test_load.py
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Starting benchmark...", flush=True)

# 1. Benchmark DocumentStore load
from services.indexing.document_store import DocumentStore
t0 = time.time()
store = DocumentStore()
count = store.load(['data/processed/processed_docs.pkl'])
t1 = time.time()
print(f"DocumentStore loaded {count} documents in {t1 - t0:.4f} seconds", flush=True)

# 2. Benchmark BM25SearchService init
from services.retrieval.bm25_search_service import BM25SearchService
t0 = time.time()
bm25 = BM25SearchService()
t1 = time.time()
print(f"BM25SearchService initialized in {t1 - t0:.4f} seconds", flush=True)

# 3. Test a quick search
t0 = time.time()
results = bm25.search("heart attack", top_k=5)
t1 = time.time()
print(f"BM25 Search completed in {t1 - t0:.4f} seconds", flush=True)
for r in results:
    print(f" - {r['doc_id']}: {r['score']} | {r['text'][:60]}...", flush=True)
