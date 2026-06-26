import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.query_processing.query_refiner import QueryRefiner

def main():
    print("Initializing QueryRefiner...")
    ref = QueryRefiner()
    
    queries_path = 'data/processed/queries.pkl'
    if not os.path.exists(queries_path):
        print(f"Error: {queries_path} not found.")
        return
        
    with open(queries_path, 'rb') as f:
        queries = pickle.load(f)
        
    for qid, qtext in list(queries.items())[:10]:
        res = ref.refine(qtext, apply_prf=False, apply_history=False, apply_profile=False)
        print("=" * 60)
        print(f"Query {qid}")
        print(f"Original: {qtext}")
        print(f"Corrected: {res['corrected']}")
        print(f"Expanded: {res['expanded_query']}")
        print(f"Synonyms: {res['synonyms_added']}")
        print(f"Corrections: {res['corrections_made']}")

if __name__ == "__main__":
    main()
