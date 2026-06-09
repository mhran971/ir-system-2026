import os

import ir_datasets
import pandas as pd
from tqdm import tqdm


def _patch_ir_datasets_text_encoding():
    """
    ir_datasets.tsv.FileLineIter uses io.TextIOWrapper without specifying an
    encoding, so on Windows it can fall back to cp1252 and fail on UTF-8 docs.
    Force UTF-8 here before any dataset iterators are created.
    """
    try:
        import ir_datasets.formats.tsv as tsv
    except Exception:
        return

    original_textio_wrapper = tsv.io.TextIOWrapper

    def utf8_text_wrapper(buffer, *args, **kwargs):
        kwargs.setdefault("encoding", "utf-8")
        kwargs.setdefault("errors", "replace")
        return original_textio_wrapper(buffer, *args, **kwargs)

    tsv.io.TextIOWrapper = utf8_text_wrapper


_patch_ir_datasets_text_encoding()

class DatasetLoader:
    def __init__(self, dataset_name="msmarco-passage/train", max_docs=None):
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
