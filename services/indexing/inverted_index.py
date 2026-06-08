from collections import defaultdict
import pickle
import os

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(list)  # term -> [doc_id, doc_id, ...]
        self.doc_freqs = {}             # term -> df
        self.doc_lengths = {}           # doc_id -> length
        self.N = 0                      # total docs
        
    def build(self, processed_docs):
        """
        processed_docs: list of dicts with 'doc_id' and 'tokens'
        """
        self.N = len(processed_docs)
        
        for doc in processed_docs:
            doc_id = doc['doc_id']
            tokens = doc['tokens']
            
            self.doc_lengths[doc_id] = len(tokens)
            
            # Track unique terms in this doc
            seen = set()
            for pos, term in enumerate(tokens):
                if term not in seen:
                    self.index[term].append(doc_id)
                    seen.add(term)
        
        # Document frequencies
        for term, postings in self.index.items():
            self.doc_freqs[term] = len(postings)
            
        print(f"Index built: {len(self.index)} terms, {self.N} docs")
    
    def save(self, path="data/index/inverted_index.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'index': dict(self.index),
                'doc_freqs': self.doc_freqs,
                'doc_lengths': self.doc_lengths,
                'N': self.N
            }, f)
        print(f"Index saved to {path}")
    
    def load(self, path="data/index/inverted_index.pkl"):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.index = defaultdict(list, data['index'])
            self.doc_freqs = data['doc_freqs']
            self.doc_lengths = data['doc_lengths']
            self.N = data['N']
    
    def get_postings(self, term):
        return self.index.get(term, [])
    
    def get_df(self, term):
        return self.doc_freqs.get(term, 0)