import sys
sys.path.append('../..')
import pickle
import os
from services.preprocessing.preprocessor import TextPreprocessor

class QueryProcessor:
    def __init__(self, preprocessor=None):
        if preprocessor is not None:
            self.preprocessor = preprocessor
        else:
            # Look for serialized preprocessor to maintain consistency
            preprocessor_paths = [
                'data/processed/preprocessor.pkl',
                '../data/processed/preprocessor.pkl',
                '../../data/processed/preprocessor.pkl',
                'data/index/preprocessor.pkl'
            ]
            loaded_preprocessor = None
            for path in preprocessor_paths:
                if os.path.exists(path):
                    try:
                        with open(path, 'rb') as f:
                            loaded_preprocessor = pickle.load(f)
                        print(f"✅ [QueryProcessor] Loaded saved preprocessor from {path}")
                        break
                    except Exception as e:
                        print(f"⚠️ [QueryProcessor] Error loading preprocessor from {path}: {e}")
            
            self.preprocessor = loaded_preprocessor or TextPreprocessor()
    
    def process(self, query_text):
        tokens = self.preprocessor.process(query_text)
        return {
            'original': query_text,
            'tokens': tokens,
            'processed': ' '.join(tokens)
        }

    def process_query(self, query_text):
        """Processes query text and returns a list of processed tokens."""
        return self.process(query_text)['tokens']