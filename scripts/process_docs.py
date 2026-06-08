import sys
sys.path.append('.')
from services.preprocessing.preprocessor import TextPreprocessor
from services.preprocessing.loader import DatasetLoader
import pandas as pd
import pickle
import os

def main():
    loader = DatasetLoader(max_docs=10000)
    df = loader.load()
    
    preprocessor = TextPreprocessor(use_stemming=False)
    
    processed = []
    for idx, row in df.iterrows():
        tokens = preprocessor.process(row['text'])
        processed.append({
            'doc_id': row['doc_id'],
            'original': row['text'],
            'tokens': tokens,
            'processed_text': ' '.join(tokens)
        })
        
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} documents")
    
    # حفظ
    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/processed_docs_10k.pkl', 'wb') as f:
        pickle.dump(processed, f)
    
    # CSV للفهرسة
    proc_df = pd.DataFrame([{k: v for k, v in d.items() if k != 'tokens'} for d in processed])
    proc_df.to_csv('data/processed/processed_docs_10k.csv', index=False)
    print("Done! Saved processed documents.")

if __name__ == "__main__":
    main()