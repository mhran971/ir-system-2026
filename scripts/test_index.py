import sys
sys.path.append('.')
import pickle
from services.indexing.inverted_index import InvertedIndex
from services.query_processing.query_processor import QueryProcessor

# 1. تحميل الوثائق المعالجة
with open('data/processed/processed_docs_10k.pkl', 'rb') as f:
    docs = pickle.load(f)

# 2. بناء الفهرس
index = InvertedIndex()
index.build(docs)
index.save()

# 3. اختبار استعلام
qp = QueryProcessor()
query = qp.process("cloud storage backup")

print(f"Query tokens: {query['tokens']}")

for term in query['tokens']:
    postings = index.get_postings(term)
    print(f"Term '{term}': {len(postings)} docs, first 5: {postings[:5]}")