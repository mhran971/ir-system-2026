import pickle
import sys
import argparse
from pathlib import Path

sys.path.append('.')

from services.indexing.inverted_index import InvertedIndex
from services.query_processing.query_processor import QueryProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Build and probe the inverted index.")
    parser.add_argument(
        "--input-dir",
        default="data/processed",
        help="Directory containing processed_docs.pkl or processed_docs_10k.pkl",
    )
    parser.add_argument(
        "--query",
        default="cloud storage backup",
        help="Query text to test against the index",
    )
    return parser.parse_args()


def load_processed_docs(input_dir):
    candidates = [
        Path(input_dir) / 'processed_docs.pkl',
        Path(input_dir) / 'processed_docs_10k.pkl',
    ]
    for path in candidates:
        if path.exists():
            with path.open('rb') as f:
                return pickle.load(f), path
    raise FileNotFoundError(f"No processed dataset found in {input_dir}")


def main():
    args = parse_args()
    print("=== Loading processed documents ===")
    docs, docs_path = load_processed_docs(args.input_dir)
    print(f"Loaded documents from {docs_path}")
    print(f"Document count: {len(docs)}")

    print("\n=== Building inverted index ===")
    index = InvertedIndex()
    index.build(docs)
    print("Index build complete")

    print("\n=== Saving index ===")
    index.save()
    print("Index saved successfully")

    print("\n=== Testing query ===")
    qp = QueryProcessor()
    query = qp.process(args.query)

    print(f"Query tokens: {query['tokens']}")

    for term in query['tokens']:
        postings = index.get_postings(term)
        print(f"Term '{term}': {len(postings)} docs -> first 5: {postings[:5]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
