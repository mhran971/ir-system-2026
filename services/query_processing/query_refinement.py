# services/query_processing/query_refinement.py

import os
import pickle
from collections import Counter
from typing import Dict, List, Tuple

from spellchecker import SpellChecker

from services.indexing.inverted_index import InvertedIndex


class QueryRefinementService:
    """
    Query Refinement Service

    Pipeline:
        1. Spell Correction
        2. Corpus-based Query Expansion

    Example:
        clud storag backp

        →

        cloud storage backup recovery files synchronization
    """

    def __init__(
        self,
        expand_terms: bool = True,
        max_expansions: int = 3,
    ):
        self.spell = SpellChecker()

        self.expand_terms = expand_terms
        self.max_expansions = max_expansions

        self.index = self._load_index()

        print(
            f"[QueryRefinementService] Ready "
            f"({len(self.index):,} terms)"
        )

    def _load_index(self) -> InvertedIndex:

        paths = [
            "data/index/inverted_index.pkl",
            "data/index/index.pkl",
        ]

        for path in paths:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    index = pickle.load(f)

                print(
                    f"[QueryRefinementService] "
                    f"Loaded index from {path}"
                )

                return index

        raise FileNotFoundError(
            "No inverted index found."
        )

    # --------------------------------------------------
    # SPELL CORRECTION
    # --------------------------------------------------

    def correct_spelling(
        self,
        query: str,
    ) -> Tuple[str, Dict[str, str]]:

        words = query.lower().split()

        misspelled = self.spell.unknown(words)

        corrections = {}
        result = []

        for word in words:

            if word in misspelled:

                candidate = self.spell.correction(word)

                if candidate and candidate != word:
                    corrections[word] = candidate
                    result.append(candidate)
                else:
                    result.append(word)

            else:
                result.append(word)

        return " ".join(result), corrections

    # --------------------------------------------------
    # QUERY EXPANSION
    # --------------------------------------------------

    def _get_related_terms(
        self,
        term: str,
    ) -> List[str]:

        if term not in self.index:
            return []

        postings = self.index[term]

        candidate_docs = list(postings.keys())[:100]

        related = Counter()

        for doc_id in candidate_docs:

            for other_term in self.index:

                if other_term == term:
                    continue

                other_postings = self.index[other_term]

                if doc_id in other_postings:
                    related[other_term] += 1

        expansions = []

        for word, _ in related.most_common(
            self.max_expansions
        ):
            expansions.append(word)

        return expansions

    def expand_query(
        self,
        query: str,
    ) -> Tuple[str, Dict[str, List[str]]]:

        if not self.expand_terms:
            return query, {}

        tokens = query.split()

        expansions = {}

        extra_terms = []

        for token in tokens:

            related = self._get_related_terms(token)

            if related:
                expansions[token] = related
                extra_terms.extend(related)

        expanded_query = query

        if extra_terms:
            expanded_query += " " + " ".join(extra_terms)

        return expanded_query, expansions

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------

    def refine(
        self,
        query: str,
    ) -> Dict:

        if not query or not query.strip():
            return {
                "original": query,
                "corrected": query,
                "expanded": query,
                "refined": query,
                "corrections": {},
                "expansions": {},
                "was_modified": False,
            }

        corrected, corrections = self.correct_spelling(
            query
        )

        expanded, expansions = self.expand_query(
            corrected
        )

        return {
            "original": query,
            "corrected": corrected,
            "expanded": expanded,
            "refined": expanded,
            "corrections": corrections,
            "expansions": expansions,
            "was_modified": bool(
                corrections or expansions
            ),
        }


if __name__ == "__main__":

    svc = QueryRefinementService(
        expand_terms=True,
        max_expansions=3,
    )

    tests = [
        "clud storag backp",
        "machine learning algoritm",
        "cloud storage backup",
    ]

    for q in tests:

        print("\n" + "=" * 60)

        result = svc.refine(q)

        print("Original   :", result["original"])
        print("Corrected  :", result["corrected"])
        print("Expanded   :", result["expanded"])

        if result["corrections"]:
            print("Corrections:", result["corrections"])

        if result["expansions"]:
            print("Expansions :", result["expansions"])