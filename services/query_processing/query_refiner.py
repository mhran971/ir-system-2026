"""
Query Refinement Service — Medical IR (ClinicalTrials)

Two public capabilities:
  refine()         → Query formulation assistance: spelling + PRF
  suggest_queries() → Query suggestion: alternative phrasings for the UI

Design decisions:
- Synonym expansion removed from refine(): even with a curated medical thesaurus
  the substring matching was adding unrelated synonyms and hurting MAP.
  PRF is strictly better — it uses terms that actually co-occur in retrieved docs.
- QuerySegmenter removed: stripping "38-year-old male" from a TREC PM query
  removes eligibility-matching terms, hurting recall on clinical trial retrieval.
- Token repetition removed: repeating base tokens 3× makes queries artificially
  long and confuses BM25's term-frequency scoring.
"""

import re
import os
import pickle
import numpy as np
from math import log
from typing import List, Dict, Set, Optional, Any, Tuple
from collections import Counter

from spellchecker import SpellChecker
from nltk.tokenize import word_tokenize

from services.query_processing.query_processor import QueryProcessor


# ── Medical thesaurus (kept for suggest_queries only) ─────────────────────────

MEDICAL_THESAURUS: Dict[str, List[str]] = {
    'cancer':           ['carcinoma', 'malignancy', 'neoplasm'],
    'breast cancer':    ['breast carcinoma', 'mammary cancer'],
    'lung cancer':      ['pulmonary carcinoma', 'lung carcinoma'],
    'colorectal cancer':['colon cancer', 'rectal cancer', 'colorectal carcinoma'],
    'pancreatic cancer':['pancreatic carcinoma', 'pancreas cancer'],
    'liver cancer':     ['hepatocellular carcinoma', 'hepatic cancer'],
    'melanoma':         ['malignant melanoma', 'cutaneous melanoma'],
    'leukemia':         ['leukaemia', 'haematological malignancy'],
    'lymphoma':         ['lymphatic cancer', 'lymphatic malignancy'],
    'tumor':            ['neoplasm', 'lesion', 'mass'],
    'metastasis':       ['metastatic disease', 'secondary tumor'],
    'mutation':         ['variant', 'alteration', 'gene mutation'],
    'amplification':    ['gene amplification', 'copy number gain'],
    'HER2':             ['ERBB2', 'HER2/neu'],
    'EGFR':             ['epidermal growth factor receptor', 'ERBB1'],
    'KRAS':             ['K-Ras', 'KRAS mutation'],
    'BRAF':             ['BRAF mutation', 'BRAF V600E'],
    'BRCA1':            ['breast cancer gene 1', 'BRCA1 mutation'],
    'BRCA2':            ['breast cancer gene 2', 'BRCA2 mutation'],
    'chemotherapy':     ['cytotoxic therapy', 'antineoplastic therapy'],
    'immunotherapy':    ['immune therapy', 'biological therapy'],
    'hypertension':     ['high blood pressure', 'HTN'],
    'diabetes':         ['diabetes mellitus', 'DM'],
}

# Clinical trial boilerplate to exclude from PRF
_PRF_EXCLUDED = {
    'patient', 'patients', 'study', 'studies', 'trial', 'trials', 'clinical',
    'treatment', 'therapy', 'therapeutic', 'evaluable', 'assessed', 'enrolled',
    'eligibility', 'criteria', 'inclusion', 'exclusion', 'date', 'group', 'groups',
    'daily', 'dose', 'doses', 'week', 'weeks', 'month', 'months', 'year', 'years',
    'day', 'days', 'history', 'diagnosed', 'diagnosis', 'active', 'prior',
    'receive', 'receiving', 'efficacy', 'safety', 'associated', 'results',
}

# Protected gene/mutation tokens — never spell-correct these
_PROTECTED_GENES = {
    'kras', 'braf', 'cdk4', 'nf2', 'akt1', 'fgfr1', 'pten', 'cdkn2a', 'nras',
    'egfr', 'eml4', 'alk', 'kit', 'pik3ca', 'brca2', 'idh1', 'stk11', 'cdk6',
    'mdm2', 'met', 'tp53', 'erbb3', 'erbb2', 'brca1', 'rb1',
}


# ── Sub-service: Medical-aware spelling correction ────────────────────────────

class MedicalSpeller:
    """
    Corrects misspelled tokens while protecting gene names, mutation codes,
    uppercase abbreviations, and numbers from being "corrected" into common
    English words (e.g., KRAS must never become "grass").
    """

    def __init__(self, known_terms: Set[str], spell_checker: SpellChecker):
        self.known_terms = known_terms
        self.spell_checker = spell_checker

    def _should_protect(self, token: str) -> bool:
        if any(c.isdigit() for c in token):
            return True
        if any(c.isupper() for c in token):
            return True
        if len(token) < 3:
            return True
        if token.lower() in _PROTECTED_GENES:
            return True
        return False

    def correct(self, query: str) -> Tuple[str, List[Tuple[str, str]]]:
        tokens = word_tokenize(query)
        out, fixes = [], []
        for tok in tokens:
            if self._should_protect(tok) or tok.lower() in self.known_terms:
                out.append(tok)
                continue
            candidate = self.spell_checker.correction(tok)
            if candidate and candidate.lower() != tok.lower() and candidate.lower() in self.known_terms:
                out.append(candidate.capitalize() if tok[0].isupper() else candidate)
                fixes.append((tok, candidate))
            else:
                out.append(tok)
        corrected = " ".join(out)
        corrected = re.sub(r'\s+([,\).])', r'\1', corrected)
        corrected = re.sub(r'(\()\s+', r'\1', corrected)
        return corrected, fixes


# ── Sub-service: Corpus-aware PRF with Semantic BERT weighting ────────────────────────────

class MedicalPRF:
    """
    Pseudo-Relevance Feedback using corpus IDF scores and semantic BERT similarity.
    Extracts high-value terms from top-retrieved documents,
    excluding clinical-trial boilerplate and boosting gene/mutation-like tokens.
    """

    def __init__(self, preprocessor, term_frequencies: Dict[str, int], total_docs: int, bert_service=None):
        self.preprocessor = preprocessor
        self.term_frequencies = term_frequencies
        self.total_docs = total_docs
        self.bert_service = bert_service  # Optional, for semantic weighting

    def extract_terms(
        self,
        top_docs: List[Dict],
        original_tokens: List[str],
        original_query: str = "",        # <--- NEW: Used for BERT semantic similarity
        num_terms: int = 3,
    ) -> List[str]:
        if not top_docs or len(top_docs) < 2:
            return []

        original_set = {t.lower() for t in original_tokens}
        term_doc_count: Counter = Counter()
        doc_term_freqs: List[Counter] = []

        # --- IMPROVEMENT 1: Use 10 documents instead of 5 ---
        for doc in top_docs[:10]:
            text = doc.get("full_text") or doc.get("text") or ""
            if not text:
                continue
            tokens = self.preprocessor.process(text)
            if not tokens:
                continue
            filtered = [
                t for t in tokens
                if t.lower() not in original_set
                and t.lower() not in _PRF_EXCLUDED
                and len(t) >= 3
            ]
            if not filtered:
                continue
            tf = Counter(filtered)
            doc_term_freqs.append(tf)
            for term in set(filtered):
                term_doc_count[term] += 1

        if not doc_term_freqs:
            return []

        N = len(doc_term_freqs)
        # --- IMPROVEMENT 2: Increase min_df to 3 to filter out noise ---
        min_df = 3
        max_df = max(int(N * 0.6), 2)

        # --- IMPROVEMENT 3: Compute BERT query vector if available ---
        query_vector = None
        if self.bert_service and original_query:
            try:
                query_vector = self.bert_service.model.encode(original_query)
            except Exception:
                query_vector = None

        scores: Dict[str, float] = {}
        for term, df in term_doc_count.items():
            if df < min_df or df > max_df:
                continue
            coll_df = self.term_frequencies.get(term, df)
            idf = log((self.total_docs + 1) / (coll_df + 1))
            boost = 2.5 if (any(c.isupper() for c in term) or any(c.isdigit() for c in term)) else 1.0
            total_tf = sum(tf.get(term, 0) for tf in doc_term_freqs)
            base_score = total_tf * idf * boost

            # --- IMPROVEMENT 4: Weight by semantic similarity to original query ---
            if query_vector is not None:
                try:
                    term_vector = self.bert_service.model.encode(term)
                    # Vectors are L2-normalized, so dot product = cosine similarity
                    sim = np.dot(query_vector, term_vector)
                    # Map similarity from [-1, 1] to [0.2, 1.2] to avoid zeroing out
                    # Terms with negative similarity get a small weight, positive get boosted
                    semantic_weight = 0.5 + 0.5 * max(0, sim)  # ranges 0.5 to 1.0
                    # Alternatively, we can use: semantic_weight = 0.5 + 0.5 * (sim + 1)/2
                    # I'll use the simpler max(0) version to heavily penalize irrelevant terms
                    base_score *= semantic_weight
                except Exception:
                    pass

            scores[term] = base_score

        return [t for t, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:num_terms]]


# ── Main orchestrator ─────────────────────────────────────────────────────────

class QueryRefiner:
    """
    Medical query refinement with two public capabilities:

    1. refine()         — formulation assistance: spelling + PRF
    2. suggest_queries() — query suggestion: clickable alternative queries for the UI

    Legacy parameters (apply_synonyms, apply_history, apply_profile, synonym_limit)
    are accepted but silently ignored so existing callers don't break.
    """

    def __init__(
        self,
        known_terms: Optional[Set[str]] = None,
        term_frequencies: Optional[Dict[str, int]] = None,
        total_docs: int = 0,
        bert_service=None,    # <--- NEW: Accept BERT service for semantic PRF
        **kwargs,             # absorbs unused legacy params (model_name, etc.)
    ):
        print("Initializing QueryRefiner...")
        self.query_processor = QueryProcessor()
        self.preprocessor = self.query_processor.preprocessor
        self.stopwords = self.preprocessor.stop_words

        self.known_terms = known_terms or set()
        self.term_frequencies = term_frequencies or {}
        self.total_docs = max(total_docs, 1)

        self.spell_checker = SpellChecker()
        self.speller = MedicalSpeller(self.known_terms, self.spell_checker)
        self.prf_service = MedicalPRF(
            self.preprocessor,
            self.term_frequencies,
            self.total_docs,
            bert_service=bert_service  # Pass it down
        )

        # Lightweight history for UI session (term counts only, no SBERT)
        self.user_history_terms: Dict[str, int] = {}

        print(f"QueryRefiner ready — vocab={len(self.known_terms):,} docs={self.total_docs:,}")

    # ── Formulation assistance ────────────────────────────────────────────────

    def refine(
        self,
        query: str,
        top_docs: Optional[List[Dict]] = None,
        apply_spelling: bool = True,
        apply_prf: bool = True,
        num_prf_terms: int = 3,
        bert_query: Optional[str] = None,   # <--- NEW: pass original query for BERT similarity
        # Legacy params — accepted but ignored
        apply_synonyms: bool = False,
        apply_history: bool = False,
        apply_profile: bool = False,
        synonym_limit: int = 0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Query formulation assistance.

        Steps:
          1. Spelling correction (medical-aware: protects genes/mutations)
          2. PRF expansion from top-retrieved documents (corpus-aware + semantic)

        Returns expanded_query = corrected_tokens + prf_terms (one occurrence each).
        Pass apply_prf=False for BERT to avoid embedding dilution.
        """
        current_query = query
        corrections_made: List[Tuple[str, str]] = []

        if apply_spelling:
            current_query, corrections_made = self.speller.correct(current_query)

        base_tokens = self.preprocessor.process(current_query)
        prf_terms: List[str] = []

        if apply_prf and top_docs and base_tokens:
            # Pass the original query (or the current one) for semantic similarity
            original_for_semantic = bert_query if bert_query else query
            prf_terms = self.prf_service.extract_terms(
                top_docs,
                base_tokens,
                original_query=original_for_semantic,
                num_terms=num_prf_terms
            )

        final_tokens = base_tokens + [t for t in prf_terms if t not in base_tokens]
        expanded_query = " ".join(final_tokens) if final_tokens else current_query

        return {
            "original":              query,
            "corrected":             current_query,
            "expanded_query":        expanded_query,
            "prf_terms_added":       prf_terms,
            "corrections_made":      corrections_made,
            # Legacy keys kept for UI/evaluate.py compatibility
            "tokens":                final_tokens,
            "weights":               {},
            "synonyms_added":        [],
            "history_boost_applied": {},
        }

    # ── Query suggestion ──────────────────────────────────────────────────────

    def suggest_queries(
        self,
        query: str,
        top_docs: Optional[List[Dict]] = None,
        n: int = 3,
    ) -> List[str]:
        """
        Generate up to n alternative query suggestions for the UI.

        Sources (in priority order):
          1. PRF terms — different subsets give different search angles
          2. Medical thesaurus — replace a key term with a clinical synonym

        Each suggestion is a complete query string ready to paste into the search box.
        """
        base_tokens = self.preprocessor.process(query)
        suggestions: List[str] = []

        # 1. PRF-based suggestions (need top_docs)
        if top_docs and len(top_docs) >= 2 and base_tokens:
            candidates = self.prf_service.extract_terms(
                top_docs, base_tokens, original_query=query, num_terms=n * 2
            )
            for term in candidates:
                if len(suggestions) >= n:
                    break
                suggestions.append(f"{query} {term}")

        # 2. Thesaurus-based suggestions (fill remaining slots)
        if len(suggestions) < n:
            for token in base_tokens:
                if len(suggestions) >= n:
                    break
                alts = MEDICAL_THESAURUS.get(token.lower()) or MEDICAL_THESAURUS.get(token)
                if not alts:
                    continue
                # Replace the token with the first thesaurus alternative
                alt = alts[0]
                pattern = re.compile(re.escape(token), re.IGNORECASE)
                new_query = pattern.sub(alt, query, count=1)
                if new_query.lower() != query.lower():
                    suggestions.append(new_query)

        return suggestions[:n]

    # ── History (lightweight, UI-only) ────────────────────────────────────────

    def update_history(self, query: str, results: Optional[List[Dict]] = None):
        tokens = self.preprocessor.process(query)
        for t in tokens:
            if len(t) > 2 and t not in self.stopwords:
                self.user_history_terms[t] = self.user_history_terms.get(t, 0) + 1

    def save_history(self, path: str = "data/user_history.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.user_history_terms, f)

    def load_history(self, path: str = "data/user_history.pkl"):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.user_history_terms = pickle.load(f)
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        return {
            "known_terms": len(self.known_terms),
            "total_docs":  self.total_docs,
            "history_len": len(self.user_history_terms),
        }