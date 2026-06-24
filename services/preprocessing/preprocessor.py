# services/preprocessing/preprocessor.py
import logging
import re
import string
from typing import List, Optional, Set

import nltk
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_wordnet_pos(tag: str) -> str:
    """Map NLTK POS tags to WordNet POS tags."""
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


class TextPreprocessor:
    """
    Text preprocessing pipeline with safe fallbacks for minimal/offline setups.
    """

    def __init__(
        self,
        language: str = "english",
        use_stemming: bool = False,
        use_lemmatization: bool = True,
        min_token_length: int = 3,
        remove_numbers: bool = True,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_stopwords: bool = True,
        custom_stopwords: Optional[Set[str]] = None,
    ):
        self.language = language
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.min_token_length = min_token_length
        self.remove_numbers = remove_numbers
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords

        try:
            self.stop_words = set(stopwords.words(language))
            logger.info("Loaded %s stop words for language %s", len(self.stop_words), language)
        except Exception as e:
            logger.warning("Could not load stop words: %s", e)
            self.stop_words = self._get_default_stopwords()

        if custom_stopwords:
            self.stop_words.update(custom_stopwords)

        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        self.stats = {
            "texts_processed": 0,
            "total_tokens": 0,
            "removed_stopwords": 0,
            "removed_short_tokens": 0,
        }

    def process(self, text: str) -> List[str]:
        if not isinstance(text, str):
            logger.warning("Input is not text: %s", type(text))
            raise ValueError(f"Expected str, got {type(text)}")

        if not text or len(text.strip()) == 0:
            return []

        try:
            if self.lowercase:
                text = text.lower()

            if self.remove_numbers:
                text = re.sub(r"\d+", " ", text)

            if self.remove_punctuation:
                text = text.translate(str.maketrans("", "", string.punctuation))

            text = re.sub(r"\s+", " ", text).strip()

            try:
                tokens = word_tokenize(text)
            except LookupError:
                tokens = re.findall(r"\b\w+\b", text)

            if self.use_lemmatization and not self.use_stemming:
                try:
                    tagged_tokens = pos_tag(tokens)
                except LookupError:
                    tagged_tokens = [(t, "") for t in tokens]
            else:
                tagged_tokens = [(t, "") for t in tokens]

            filtered_tagged_tokens = []
            for token, tag in tagged_tokens:
                if len(token) < self.min_token_length:
                    self.stats["removed_short_tokens"] += 1
                    continue

                if self.remove_stopwords and token in self.stop_words:
                    self.stats["removed_stopwords"] += 1
                    continue

                filtered_tagged_tokens.append((token, tag))

            if self.use_stemming:
                processed_tokens = [self._stem(t[0]) for t in filtered_tagged_tokens]
            elif self.use_lemmatization:
                processed_tokens = [
                    self._lemmatize(word, get_wordnet_pos(tag))
                    for word, tag in filtered_tagged_tokens
                ]
            else:
                processed_tokens = [t[0] for t in filtered_tagged_tokens]

            self.stats["texts_processed"] += 1
            self.stats["total_tokens"] += len(processed_tokens)
            return processed_tokens
        except Exception as e:
            logger.error("Error processing text: %s", e)
            raise

    def process_batch(self, texts: List[str]) -> List[List[str]]:
        logger.info("Starting processing of %s texts", len(texts))
        results = []

        for i, text in enumerate(texts):
            try:
                results.append(self.process(text))
                if (i + 1) % 1000 == 0:
                    logger.info("Processed %s/%s texts", i + 1, len(texts))
            except Exception as e:
                logger.warning("Error in text #%s: %s", i, e)
                results.append([])

        logger.info("Completed processing of %s texts", len(texts))
        return results

    def _stem(self, word: str) -> str:
        try:
            return self.stemmer.stem(word)
        except Exception as e:
            logger.warning("Error in stemming: %s", e)
            return word

    def _lemmatize(self, word: str, pos: Optional[str] = None) -> str:
        try:
            if pos:
                return self.lemmatizer.lemmatize(word, pos)
            return self.lemmatizer.lemmatize(word)
        except Exception as e:
            logger.warning("Error in lemmatization: %s", e)
            return word

    def _get_default_stopwords(self) -> Set[str]:
        return {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
            "to", "was", "were", "will", "with", "i", "you", "we", "they",
            "this", "that", "these", "those", "am", "do", "does", "did",
            "doing", "have", "having", "or", "but", "not", "so", "such",
            "can", "could", "would", "should", "may", "might", "must",
            "up", "down", "out", "off", "over", "under", "again", "further",
            "then", "once", "here", "there", "all", "any", "both", "each",
            "few", "more", "most", "other", "some", "no", "nor", "only",
            "own", "same", "than", "too", "very", "just",
        }

    def get_stats(self) -> dict:
        return {
            "texts_processed": self.stats["texts_processed"],
            "total_tokens": self.stats["total_tokens"],
            "removed_stopwords": self.stats["removed_stopwords"],
            "removed_short_tokens": self.stats["removed_short_tokens"],
            "avg_tokens_per_text": (
                self.stats["total_tokens"] / self.stats["texts_processed"]
                if self.stats["texts_processed"] > 0
                else 0
            ),
        }

    def print_stats(self) -> None:
        stats = self.get_stats()
        print("\nProcessing statistics:")
        print("=" * 50)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        print("=" * 50)


SimplePreprocessor = TextPreprocessor
