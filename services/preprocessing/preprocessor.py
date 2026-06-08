import nltk
import spacy
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

class TextPreprocessor:
    def __init__(self, language='english', use_stemming=False):
        self.language = language
        self.use_stemming = use_stemming
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words(language))
        self.nlp = spacy.load('en_core_web_sm')
        
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