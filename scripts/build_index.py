import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
from collections import defaultdict


class SimpleIndex:
    def __init__(self):
        self.index = defaultdict(list)

    def build(self, documents):
        print(f"Building index for {len(documents)} docs...")
        for i, doc in enumerate(documents):
            doc_id = doc['doc_id']
            tokens = doc['tokens']

            term_freq = {}
            for term in tokens:
                term_freq[term] = term_freq.get(term, 0) + 1

            for term, freq in term_freq.items():
                self.index[term].append((doc_id, freq))

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1} docs")

        print(f"Done! Total unique terms: {len(self.index)}")

    def save(self, path='data/index/index.pkl'):
        os.makedirs('data/index', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(dict(self.index), f)
        print(f"Saved to {path}")


def main():
    # Try both 5000 and 200000 doc files automatically
    candidates = [
        'data/processed/processed_docs_200000.pkl',
        'data/processed/processed_docs_5000.pkl',
        'data/processed/processed_docs.pkl',
    ]

    input_file = None
    for path in candidates:
        if os.path.exists(path):
            input_file = path
            break

    if not input_file:
        print("❌ No processed docs file found.")
        print("   Run: python scripts/process_docs_new.py  first.")
        return

    print(f"📂 Loading: {input_file}")
    with open(input_file, 'rb') as f:
        docs = pickle.load(f)

    print(f"   Loaded {len(docs)} documents")

    index = SimpleIndex()
    index.build(docs)
    index.save()

    print("\n✅ Index built successfully!")
    print("   Next step: python scripts/build_bert_index.py")


if __name__ == "__main__":
    main()