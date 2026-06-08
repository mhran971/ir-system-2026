import sys
sys.path.append('../..')
from services.preprocessing.preprocessor import TextPreprocessor

class QueryProcessor:
    def __init__(self, preprocessor=None):
        self.preprocessor = preprocessor or TextPreprocessor()
    
    def process(self, query_text):
        tokens = self.preprocessor.process(query_text)
        return {
            'original': query_text,
            'tokens': tokens,
            'processed': ' '.join(tokens)
        }