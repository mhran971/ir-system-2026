import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

class TextPreprocessor:
    def __init__(self, language='english', use_stemming=False):
        self.language = language
        self.use_stemming = use_stemming
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self._ensure_nltk_resources()
        self.stop_words = set(stopwords.words(language))
        self.nlp = self._load_spacy_model()

    def _ensure_nltk_resources(self):
        resources = [
            ('corpora/stopwords', 'stopwords'),
            ('tokenizers/punkt', 'punkt'),
            ('corpora/wordnet', 'wordnet'),
            ('corpora/omw-1.4', 'omw-1.4'),
        ]
        for resource_path, package_name in resources:
            try:
                nltk.data.find(resource_path)
            except LookupError:
                nltk.download(package_name, quiet=True)

    def _load_spacy_model(self):
        try:
            import spacy
            return spacy.load('en_core_web_sm')
        except Exception:
            return None
        
    def normalize(self, text):
        return text.lower().strip()
    
    def remove_punctuation(self, text):
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def tokenize(self, text):
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        return [t for t in tokens if t not in self.stop_words and len(t) > 2]
    
    def stem(self, tokens):
        return [self.stemmer.stem(t) for t in tokens]
    
    def lemmatize(self, tokens):
        return [self.lemmatizer.lemmatize(t) for t in tokens]
    
    def process(self, text):
        text = self.normalize(text)
        text = self.remove_punctuation(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        if self.use_stemming:
            tokens = self.stem(tokens)
        else:
            tokens = self.lemmatize(tokens)
        return tokens
