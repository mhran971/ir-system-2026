import pickle
import os
from pathlib import Path

from services.indexing.inverted_index import InvertedIndex
from services.query_processing.query_processor import QueryProcessor


def load_processed_docs():
    input_dir = os.environ.get("IR_PROCESSED_DIR", "data/processed")
    candidates = [
        Path(input_dir) / 'processed_docs.pkl',
        Path(input_dir) / 'processed_docs_10k.pkl',
    ]
    for path in candidates:
        if path.exists():
            with path.open('rb') as f:
                return pickle.load(f)
    raise FileNotFoundError(f"No processed dataset found in {input_dir}")


def test_index_builds_and_queries():
    docs = load_processed_docs()
    index = InvertedIndex()
    index.build(docs)

    qp = QueryProcessor()
    query = qp.process("cloud storage backup")

    assert query["tokens"]
    for term in query["tokens"]:
        postings = index.get_postings(term)
        assert isinstance(postings, list)
