# services/ranking/embeddings/bert_search_service.py
"""
BERT Embedding Search Service.

Responsible for:
- Building BERT/SBERT document vectors from processed documents
- Storing vectors in FAISS for fast retrieval
- Searching by encoding a query and finding nearest neighbors
- Returning ranked results with document text

This is ONE service in the SOA architecture.
It is completely independent from VSM-TF-IDF and BM25.
"""

import os
import sys
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.append(str(Path(__file__).resolve().parents[3]))

from services.preprocessing.preprocessor import TextPreprocessor
from services.query_processing.query_processor import QueryProcessor
from .embedding_model import EmbeddingModel
from .vector_store import VectorStore


# ── Paths ─────────────────────────────────────────────────────────────────────
_PROCESSED_DATA_PATHS = [
    "data/processed/processed_docs_200000.pkl",
    "data/processed/processed_docs_5000.pkl",
    "data/processed/processed_docs.pkl",
]

_VECTOR_STORE_PATH = "data/index/bert_vector_store.faiss"
_DOC_TEXT_PATH     = "data/index/bert_doc_texts.pkl"


class BERTSearchService:
    """
    BERT-based semantic search service.

    Encodes documents with Sentence-BERT, stores vectors in FAISS,
    and retrieves the most semantically similar documents for a query.

    Usage:
        service = BERTSearchService()
        results = service.search("cloud storage backup", top_k=10)
    """

    def __init__(self, model_key: str = "fast", force_rebuild: bool = False):
        self.query_processor = QueryProcessor()
        self.preprocessor  = self.query_processor.preprocessor
        self.model         = EmbeddingModel(model_key=model_key)
        self.vector_store: Optional[VectorStore] = None
        self._doc_texts:   Dict[str, str] = {}
        self.total_docs    = 0

        self._initialize(force_rebuild)

    # ── Initialization ────────────────────────────────────────────────────────

    def _initialize(self, force_rebuild: bool) -> None:
        if not force_rebuild and os.path.exists(_VECTOR_STORE_PATH):
            self._load_index()
        else:
            print("🔨 Building BERT index from scratch...")
            docs = self._load_processed_documents()
            if not docs:
                raise RuntimeError(
                    "No processed documents found.\n"
                    "Run: python scripts/process_docs_new.py  first."
                )
            self._build_index(docs)
            self._save_index()

        self.total_docs = len(self._doc_texts)
        print(f"\n✅ BERTSearchService ready — {self.total_docs} documents indexed")

    def _load_processed_documents(self) -> List[Dict]:
        for path in _PROCESSED_DATA_PATHS:
            if os.path.exists(path):
                print(f"📂 Loading documents from {path}")
                with open(path, "rb") as f:
                    docs = pickle.load(f)
                print(f"   Loaded {len(docs)} documents")
                return docs
        return []

    def _build_index(self, docs: List[Dict]) -> None:
        """
        Encode all documents and store in FAISS.

        Supports docs with any of these text fields (in priority order):
            'text'  → original MS MARCO passage  ← added by updated process_docs_new.py
            'original'
            'processed_text' → fallback (cleaned tokens joined)
        """
        print(f"\n🔨 Encoding {len(docs)} documents with BERT...")
        print("   (This runs once and is saved to disk for future use)")

        doc_ids          = []
        texts_to_encode  = []

        for doc in docs:
            doc_id = str(doc.get("doc_id", ""))

            # ── FIX: try all possible text field names ──────────────────────
            text = (
                doc.get("text")             # original passage (preferred)
                or doc.get("original")      # alternative field name
                or doc.get("processed_text")# fallback: cleaned tokens
                or ""
            )
            # ────────────────────────────────────────────────────────────────

            if not doc_id or not text:
                continue

            doc_ids.append(doc_id)
            texts_to_encode.append(text)
            self._doc_texts[doc_id] = text   # store for result display

        print(f"   Encoding {len(doc_ids)} documents...")

        vectors = self.model.encode_batch(
            texts_to_encode,
            batch_size=64,
            show_progress=True
        )

        self.vector_store = VectorStore(dim=self.model.dim)
        self.vector_store.add_documents(doc_ids, vectors)

        print(f"✅ Encoded and indexed {len(doc_ids)} documents")

    def _save_index(self) -> None:
        self.vector_store.save(_VECTOR_STORE_PATH)
        os.makedirs(os.path.dirname(_DOC_TEXT_PATH), exist_ok=True)
        with open(_DOC_TEXT_PATH, "wb") as f:
            pickle.dump(self._doc_texts, f)
        print(f"✅ Doc texts saved → {_DOC_TEXT_PATH}")

    def _load_index(self) -> None:
        print("📂 Loading pre-built BERT index from disk...")
        self.vector_store = VectorStore.load(_VECTOR_STORE_PATH)
        with open(_DOC_TEXT_PATH, "rb") as f:
            self._doc_texts = pickle.load(f)
        print(f"   Loaded {len(self._doc_texts)} document texts")

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Find the most semantically similar documents for a query.

        Args:
            query: User search query string
            top_k: Number of results to return

        Returns:
            List of dicts: doc_id, score, text, full_text, method
        """
        if self.vector_store is None or len(self.vector_store) == 0:
            print("❌ No vector index loaded")
            return []

        if not query or not query.strip():
            return []

        print(f"\n🔍 BERT Search: '{query}'")

        query_vector = self.model.encode(query)
        raw_results  = self.vector_store.search(query_vector, top_k=top_k)
        print(f"   Found {len(raw_results)} candidates")

        preview_len = 300
        results     = []

        for doc_id, score in raw_results:
            text    = self._doc_texts.get(doc_id, "")
            preview = text[:preview_len] + ("..." if len(text) > preview_len else "")

            results.append({
                "doc_id":    doc_id,
                "score":     round(score, 6),
                "text":      preview,
                "full_text": text,
                "method":    f"BERT ({self.model.model_name})",
            })

        print(f"✅ Returning {len(results)} results")
        return results

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_documents": self.total_docs,
            "model_name":      self.model.model_name,
            "vector_dim":      self.model.dim,
            "vectors_stored":  len(self.vector_store) if self.vector_store else 0,
        }


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("🔍 BERT Search Service — Standalone Test")
    print("=" * 60)

    service = BERTSearchService(model_key="fast")

    stats = service.get_stats()
    print(f"\n📊 Stats:")
    for k, v in stats.items():
        print(f"   {k}: {v}")

    while True:
        query = input("\n🔍 Enter query (or 'quit'): ").strip()
        if query.lower() == "quit":
            break

        results = service.search(query, top_k=5)
        if results:
            for i, r in enumerate(results, 1):
                print(f"\n{i}. [{r['score']:.4f}] doc {r['doc_id']}")
                print(f"   {r['text'][:200]}")
        else:
            print("   No results found.")