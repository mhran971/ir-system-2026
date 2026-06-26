import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.ranking.embeddings.bert_search_service import BERTSearchService
from services.ranking.embeddings.embedding_model import EmbeddingModel


def main():
    parser = argparse.ArgumentParser(description="Build the BERT / SBERT FAISS index.")
    parser.add_argument(
        "--model", "-m",
        default=os.getenv("BERT_MODEL_KEY", "fast"),
        choices=list(EmbeddingModel.MODELS.keys()),
        help=(
            "Which embedding model to use (default: fast).\n"
            "  fast     → multi-qa-MiniLM-L6-cos-v1 (384-dim, retrieval-optimized)\n"
            "  balanced → multi-qa-mpnet-base-dot-v1 (768-dim, best quality)\n"
            "  clinical → pritamdeka/S-PubMedBert-MS-MARCO (768-dim, biomedical)\n"
            "  multilingual → paraphrase-multilingual-MiniLM-L12-v2\n"
            "You can also set BERT_MODEL_KEY env variable instead of passing --model."
        ),
    )
    args = parser.parse_args()

    model_name = EmbeddingModel.MODELS.get(args.model, args.model)

    print("=" * 60)
    print("Building BERT Vector Index")
    print("=" * 60)
    print(f"\nModel key : {args.model}")
    print(f"Model name: {model_name}")
    print("\nThis will:")
    print("  1. Load processed documents (data/processed/processed_docs.pkl)")
    print("  2. Encode each doc using bert_text (title + condition + summary)")
    print("     — falls back to full 'text' field if bert_text is absent")
    print("  3. Save FAISS index to data/index/")
    print("\n  If bert_text field is missing, re-run process_docs.py first:")
    print("    python scripts/process_docs.py\n")
    print("  Encoding 241 K documents takes ~10-40 min depending on hardware.\n")

    service = BERTSearchService(model_key=args.model, force_rebuild=True)

    stats = service.get_stats()
    print("\n" + "=" * 60)
    print("BERT Index Built Successfully!")
    print("=" * 60)
    print(f"   Documents indexed : {stats['total_documents']}")
    print(f"   Model             : {stats['model_name']}")
    print(f"   Vector dimension  : {stats['vector_dim']}")
    print(f"   Vectors stored    : {stats['vectors_stored']}")
    print("\nNext steps:")
    print("  Evaluate : python scripts/evaluate.py")
    print("  UI       : streamlit run ui/app.py")


if __name__ == "__main__":
    main()
