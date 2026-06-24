import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.retrieval.vsm_search_service import VSMSearchService
from services.retrieval.bm25_search_service import BM25SearchService
from services.ranking.embeddings.bert_search_service import BERTSearchService

def test_service(service, name, query):
    print(f"\n--- Testing {name} for query: '{query}' ---")
    try:
        results = service.search(query, top_k=2)
        print(f"Results count: {len(results)}")
        for idx, result in enumerate(results):
            print(f"Result {idx+1}:")
            print(f"  doc_id   : {result.get('doc_id')}")
            print(f"  score    : {result.get('score')}")
            print(f"  text     : {result.get('text')[:60]}...")
            print(f"  full_text: {result.get('full_text')[:60]}...")
            print(f"  method   : {result.get('method')}")
            
            # Assert keys
            for key in ['doc_id', 'score', 'text', 'full_text', 'method']:
                assert key in result, f"Missing key {key} in result of {name}"
        print(f"✅ {name} passed basic checks!")
    except Exception as e:
        print(f"❌ {name} failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("Pre-loading DocumentStore with 5,000 documents to save memory...")
    from services.indexing.document_store import DocumentStore
    ds = DocumentStore()
    ds.load(['data/processed/processed_docs_5000.pkl'])

    print("Initializing services...")
    vsm = VSMSearchService()
    bm25 = BM25SearchService()
    
    # Initialize BERT (only rebuild if vector store is missing)
    bert = BERTSearchService(model_key="fast", force_rebuild=False)
    
    query = "cloud storage backup"
    test_service(vsm, "VSM", query)
    test_service(bm25, "BM25", query)
    test_service(bert, "BERT", query)
    
    # Test empty query handling
    print("\n--- Testing empty/whitespace query ---")
    for service, name in [(vsm, "VSM"), (bm25, "BM25"), (bert, "BERT")]:
        res = service.search("   ")
        print(f"{name} empty query result: {res}")
        assert res == [], f"{name} did not return [] for empty/whitespace query"
    print("✅ All services correctly returned [] for empty/whitespace queries!")

if __name__ == "__main__":
    main()
