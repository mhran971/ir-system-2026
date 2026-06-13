import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.ranking.embeddings.bert_search_service import BERTSearchService


def main():
    print("=" * 60)
    print("🔨 Building BERT Vector Index")
    print("=" * 60)
    print("\nThis will:")
    print("  1. Load your processed documents")
    print("  2. Encode them with Sentence-BERT (all-MiniLM-L6-v2)")
    print("  3. Save the FAISS index to data/index/")
    print("\n⚠️  This takes a few minutes the FIRST time.")
    print("     After that, the index loads instantly from disk.\n")

    # force_rebuild=True → always rebuild fresh when running this script
    service = BERTSearchService(model_key="fast", force_rebuild=True)

    stats = service.get_stats()
    print("\n" + "=" * 60)
    print("✅ BERT Index Built Successfully!")
    print("=" * 60)
    print(f"   Documents indexed : {stats['total_documents']}")
    print(f"   Model             : {stats['model_name']}")
    print(f"   Vector dimension  : {stats['vector_dim']}")
    print(f"   Vectors stored    : {stats['vectors_stored']}")
    print("\n🚀 You can now run: streamlit run ui/app.py")


if __name__ == "__main__":
    main()