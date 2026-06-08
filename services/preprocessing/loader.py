import ir_datasets
import pandas as pd
from tqdm import tqdm
import os
import json

class DatasetLoader:
    def __init__(self, dataset_name="msmarco-passage/dev/small", max_docs=None):
        self.dataset_name = dataset_name
        self.max_docs = max_docs
        
    def load(self):
        print(f"Loading dataset: {self.dataset_name}")
        dataset = ir_datasets.load(self.dataset_name)
        
        docs = []
        for i, doc in enumerate(tqdm(dataset.docs_iter())):
            if self.max_docs and i >= self.max_docs:
                break
            docs.append({
                'doc_id': doc.doc_id,
                'text': doc.text
            })
        
        df = pd.DataFrame(docs)
        print(f"Loaded {len(df)} documents")
        return df
    
    def save_raw(self, df, path="data/raw/documents.csv"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Saved to {path}")
    
    def load_queries(self):
        dataset = ir_datasets.load(self.dataset_name)
        queries = []
        for q in dataset.queries_iter():
            queries.append({'query_id': q.query_id, 'text': q.text})
        return pd.DataFrame(queries)
    
    def load_qrels(self):
        dataset = ir_datasets.load(self.dataset_name)
        qrels = []
        for qrel in dataset.qrels_iter():
            qrels.append({
                'query_id': qrel.query_id,
                'doc_id': qrel.doc_id,
                'relevance': qrel.relevance
            })
        return pd.DataFrame(qrels)