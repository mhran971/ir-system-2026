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

    def process_query(self, query_text):
        """Processes query text and returns a list of processed tokens."""
        return self.process(query_text)['tokens']