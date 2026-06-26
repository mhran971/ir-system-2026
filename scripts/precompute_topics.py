# scripts/precompute_topics.py
"""
Precomputes medical topics for all 241,006 documents in ClinicalTrials 2017.
This script is run once to set up the classification tables.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.clustering.clinical_clustering import ClinicalTrialsClusterer

def main():
    print("=" * 60)
    print("🧠 ClinicalTrials Medical Topic Precomputation")
    print("=" * 60)
    
    db_path = 'data/processed/doc_store.db'
    if not os.path.exists(db_path):
        print(f"❌ Database not found at {db_path}.")
        print("   Please make sure the processed SQLite database exists.")
        return

    clusterer = ClinicalTrialsClusterer(db_path=db_path)
    
    # Run precomputation (force=True to make sure it runs and updates everything)
    clusterer.precompute_all_documents(force=True)
    
    # Get global statistics
    counts = clusterer.get_global_counts()
    print("\n" + "=" * 60)
    print("📊 GLOBAL TOPIC STATISTICS")
    print("=" * 60)
    
    for category_name, keyword_dict in clusterer.CATEGORY_MAPPING.items():
        label = clusterer.CATEGORY_LABELS[category_name]
        print(f"\n{label}:")
        category_counts = counts.get(category_name, {})
        
        # Sort by count desc
        sorted_counts = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        for topic, count in sorted_counts:
            # Capitalize topic
            print(f"   - {topic.upper():<20} : {count:,} docs")

    print("\n✅ Precomputation and validation complete!")

if __name__ == "__main__":
    main()
