# scripts/build_bm25_index.py
"""
Verify and output statistics for the BM25 retrieval system.
Uses the unified global InvertedIndex and DocumentStore.

Run:
    python scripts/build_bm25_index.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.retrieval.bm25_search_service import BM25SearchService

def main():
    print("=" * 60)
    print("📊 Verifying BM25 Search System and Core Index")
    print("=" * 60)
    
    # Initialize BM25 Search Service (which loads unified DocumentStore and InvertedIndex)
    print("\n[1/2] Loading unified InvertedIndex and DocumentStore...")
    try:
        bm25_service = BM25SearchService(k1=1.5, b=0.75)
    except Exception as e:
        print(f"❌ Error loading system: {e}")
        print("   Please run first: python scripts/build_index.py")
        return
        
    # Print statistics
    stats = bm25_service.get_stats()
    print("\n" + "=" * 60)
    print("📊 BM25 Search System Statistics")
    print("=" * 60)
    print(f"   Total documents   : {stats['total_documents']:,}")
    print(f"   Unique terms      : {stats['unique_terms']:,}")
    print(f"   Total postings    : {bm25_service.inverted_index.total_entries:,}")
    print(f"   Avg doc length    : {stats['avg_doc_length']:.2f}")
    print(f"   Parameters        : k1={bm25_service.k1}, b={bm25_service.b}")
    
    print("\n✅ BM25 search system is ready and verified!")
    print("\n🚀 Next steps:")
    print("   - Run interface: streamlit run ui/app.py")


if __name__ == "__main__":
    main()