"""
Query Refinement Service – 4-Stage Pipeline (Optimized)
1. Fast Spelling Correction
2. Context-Filtered Synonym Expansion
3. Pseudo-Relevance Feedback (PRF)
4. Personalized History Weighting
"""

import re
import os
import pickle
from math import log
from typing import List, Dict, Set, Optional, Any
from collections import Counter

from spellchecker import SpellChecker
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.tokenize import word_tokenize

from services.query_processing.query_processor import QueryProcessor
from services.preprocessing.preprocessor import get_wordnet_pos


class QueryRefiner:

    def __init__(
        self,
        known_terms: Optional[Set[str]] = None,
        term_frequencies: Optional[Dict[str, int]] = None,
        total_docs: int = 0
    ):
        """
        Parameters
        ----------
        known_terms
            Vocabulary extracted from inverted index
        term_frequencies
            Document frequency of each term
        total_docs
            Total documents in collection
        """
        print("🔧 Initializing QueryRefiner...")

        self.query_processor = QueryProcessor()
        self.preprocessor = self.query_processor.preprocessor

        self.spell_checker = SpellChecker()

        self.stopwords = self.preprocessor.stop_words

        self.known_terms = known_terms or set()
        self.term_frequencies = term_frequencies or {}

        self.total_docs = max(total_docs, 1)

        self.user_history_terms: Dict[str, float] = {}

        print(
            f"✅ QueryRefiner Ready "
            f"(Terms={len(self.known_terms):,}, Docs={self.total_docs:,})"
        )

    # ============================================================
    # STAGE 1
    # SPELLING CORRECTION
    # ============================================================

    def correct_spelling(self, query: str) -> str:
        raw_tokens = word_tokenize(query)
        corrected_tokens = []

        for token in raw_tokens:
            if (
                token.lower() in self.known_terms
            ):
                corrected_tokens.append(token)
                continue

            if (
                not re.match(r'^[a-zA-Z]+$', token)
                or len(token) < 3
            ):
                corrected_tokens.append(token)
                continue

            candidate = self.spell_checker.correction(token)

            if (
                candidate
                and candidate != token
            ):
                if (
                    self.known_terms
                    and candidate.lower() in self.known_terms
                ):
                    if token[0].isupper():
                        candidate = candidate.capitalize()
                    corrected_tokens.append(candidate)
                else:
                    corrected_tokens.append(token)
            else:
                corrected_tokens.append(token)

        return " ".join(corrected_tokens)

    # ============================================================
    # STAGE 2
    # SYNONYM EXPANSION
    # ============================================================

    def expand_synonyms(
        self,
        tokens: List[str],
        max_synonyms: int = 2
    ) -> List[str]:
        expanded = list(tokens)

        if not tokens:
            return expanded

        if not self.known_terms:
            return expanded

        try:
            tagged_tokens = pos_tag(tokens)
        except Exception:
            tagged_tokens = [(t, "NN") for t in tokens]

        expanded_lower = {t.lower() for t in expanded}

        BLOCKED_SYNONYM_TOKENS = {
            'cancer', 'tumor', 'tumour', 'disease', 'none', 'male', 'female', 
            'patient', 'patients', 'year', 'yearold', 'month', 'monthold', 
            'day', 'dayold', 'old', 'age', 'history', 'symptom', 'symptoms',
            'therapy', 'treatment', 'inactivating', 'inactivate', 'active',
            'activation', 'amplification', 'amplify', 'loss', 'gain', 'mutation',
            'mutate', 'mutated', 'variant', 'gene', 'protein', 'receptor',
            'kinase', 'inhibitor', 'blocker', 'depression', 'hypertension',
            'diabetes'
        }

        for token, pos in tagged_tokens:
            if token.lower() in BLOCKED_SYNONYM_TOKENS:
                continue
            wn_pos = get_wordnet_pos(pos) or wordnet.NOUN
            synsets = wordnet.synsets(token, pos=wn_pos)
            added = 0

            for synset in synsets:
                if added >= max_synonyms:
                    break

                for lemma in synset.lemmas():
                    synonym = lemma.name().replace("_", " ")

                    if (
                        synonym.lower() == token.lower()
                        or synonym.lower() in self.stopwords
                        or len(synonym) < 3
                        or len(synonym) > 25
                        or len(synonym.split()) > 2
                        or not re.match(r'^[a-zA-Z\s]+$', synonym)
                    ):
                        continue

                    df = self.term_frequencies.get(synonym.lower(), 0)

                    if (
                        synonym.lower() in self.known_terms
                        and synonym.lower() not in expanded_lower
                        and df >= 5
                    ):
                        expanded.append(synonym)
                        expanded_lower.add(synonym.lower())
                        added += 1
                        break

        return expanded

    # ============================================================
    # STAGE 3
    # PSEUDO RELEVANCE FEEDBACK
    # ============================================================

    def extract_prf_terms(
        self,
        top_docs: List[Dict],
        original_tokens: List[str],
        num_terms: int = 3
    ) -> List[str]:
        if not top_docs:
            return []

        if len(top_docs) < 2:
            return []

        N = min(len(top_docs), 5)

        doc_term_freqs = []
        term_doc_count = Counter()

        for doc in top_docs[:5]:
            text = doc.get("full_text", doc.get("text", ""))
            if not text:
                continue

            tokens = self.preprocessor.process(text)
            if not tokens:
                continue

            tf_counter = Counter(tokens)
            doc_term_freqs.append(tf_counter)

            for term in set(tokens):
                if term not in self.stopwords and len(term) > 2:
                    term_doc_count[term] += 1

        if not doc_term_freqs:
            return []

        scores = {}
        original_set = set(original_tokens)

        min_df = 2
        max_df = max(int(N * 0.8), 2)

        for term, df in term_doc_count.items():
            if term in original_set:
                continue

            if df < min_df:
                continue

            if df > max_df:
                continue

            collection_df = self.term_frequencies.get(term, df)

            idf = log((self.total_docs + 1) / (collection_df + 1))

            total_tf = sum(tf.get(term, 0) for tf in doc_term_freqs)

            scores[term] = total_tf * idf

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [term for term, _ in ranked[:num_terms]]

    # ============================================================
    # STAGE 4
    # HISTORY
    # ============================================================

    def update_history(self, query: str):
        tokens = self.preprocessor.process(query)

        for token in tokens:
            if len(token) > 2 and token not in self.stopwords:
                self.user_history_terms[token] = self.user_history_terms.get(token, 0) + 1

    def get_history_weights(self, tokens: List[str]) -> Dict[str, float]:
        weights = {}

        for token in tokens:
            if token in self.user_history_terms:
                weights[token] = log(1 + self.user_history_terms[token])

        return weights

    # ============================================================
    # MAIN ORCHESTRATOR
    # ============================================================

    def refine(
        self,
        query: str,
        apply_spelling: bool = True,
        apply_synonyms: bool = True,
        apply_prf: bool = True,
        apply_history: bool = True,
        top_docs: Optional[List[Dict]] = None,
        num_prf_terms: int = 3,
        synonym_limit: int = 2
    ) -> Dict[str, Any]:
        print(f"\n🔧 Refining query: {query}")

        current_query = query

        if apply_spelling:
            current_query = self.correct_spelling(current_query)
            print(f"   ✏️ Spelling: '{query}' → '{current_query}'")

        result = {
            "original": query,
            "corrected": current_query,
            "tokens": [],
            "weights": {},
            "expanded_query": "",
            "prf_terms_added": [],
            "synonyms_added": [],
            "history_boost_applied": {}
        }

        base_tokens = self.preprocessor.process(current_query)

        if not base_tokens:
            result["expanded_query"] = current_query
            return result

        final_tokens = list(base_tokens)

        # -------------------------
        # Synonyms
        # -------------------------
        synonyms_added = []

        if apply_synonyms:
            expanded_tokens = self.expand_synonyms(base_tokens, max_synonyms=synonym_limit)
            base_set = set(base_tokens)

            for token in expanded_tokens:
                if token not in base_set and token not in final_tokens:
                    final_tokens.append(token)
                    synonyms_added.append(token)

        # -------------------------
        # PRF
        # -------------------------
        prf_terms = []

        if apply_prf and top_docs:
            prf_terms = self.extract_prf_terms(top_docs, base_tokens, num_terms=num_prf_terms)

            for token in prf_terms:
                if token not in final_tokens:
                    final_tokens.append(token)

        # -------------------------
        # History
        # -------------------------
        history_boost = {}

        if apply_history and self.user_history_terms:
            history_boost = self.get_history_weights(base_tokens)

        # -------------------------
        # Build Weights
        # -------------------------
        weights = {token: 1.0 for token in final_tokens}

        for token, boost in history_boost.items():
            weights[token] = weights.get(token, 1.0) + boost

        for token in prf_terms:
            if token in weights:
                weights[token] *= 0.6

        for token in synonyms_added:
            if token in weights:
                weights[token] *= 0.3

        for token in base_tokens:
            weights[token] = max(weights.get(token, 1.0), 1.0)

        # -------------------------
        # Weighted Query Expansion
        # -------------------------
        weighted_tokens = []

        for token in final_tokens:
            weight = weights.get(token, 1.0)

            if weight >= 2.5:
                repeats = 3
            elif weight >= 1.5:
                repeats = 2
            else:
                repeats = 1

            weighted_tokens.extend([token] * repeats)

        expanded_query = " ".join(weighted_tokens)

        result["tokens"] = final_tokens
        result["weights"] = weights
        result["expanded_query"] = expanded_query
        result["prf_terms_added"] = prf_terms
        result["synonyms_added"] = synonyms_added
        result["history_boost_applied"] = history_boost

        print(f"   📚 Synonyms: {synonyms_added}")
        print(f"   🔥 PRF: {prf_terms}")
        print(f"   📜 History: {history_boost}")
        print(f"   ✅ Tokens: {final_tokens}")

        return result

    # ============================================================
    # SAVE / LOAD HISTORY
    # ============================================================

    def save_history(self, path: str = "data/user_history.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self.user_history_terms, f)

    def load_history(self, path: str = "data/user_history.pkl"):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.user_history_terms = pickle.load(f)