# services/clustering/clustering_service.py
"""
Clustering Service for dynamic search results clustering.
Groups retrieved documents into clusters using their SBERT vectors and KMeans,
reduces dimensionality to 2D with PCA for visualization, and extracts key terms
to dynamically label each cluster.
"""

import collections
import re
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Medical and generic English stop words to filter out from cluster labels
STOP_WORDS = {
    'the', 'of', 'and', 'in', 'to', 'a', 'is', 'for', 'with', 'on', 'as', 'by', 'an', 'at', 'this', 'that', 'from',
    'or', 'but', 'are', 'was', 'be', 'were', 'clinical', 'trial', 'trials', 'patient', 'patients', 'treatment',
    'study', 'results', 'data', 'disease', 'diseases', 'group', 'groups', 'therapy', 'therapies', 'effect',
    'effects', 'associated', 'using', 'used', 'use', 'both', 'between', 'during', 'after', 'before', 'has', 'have',
    'had', 'been', 'which', 'who', 'its', 'their', 'there', 'they', 'not', 'we', 'our', 'out', 'other', 'some',
    'no', 'any', 'first', 'two', 'new', 'years', 'age', 'vs', 'compared', 'primary', 'secondary', 'outcome',
    'outcomes', 'rate', 'rates', 'dose', 'doses', 'dose-limiting', 'safety', 'efficacy', 'evaluate', 'investigate',
    'showed', 'significantly', 'statistical', 'statistically', 'p', 'value', 'months', 'weeks', 'days', 'cohort',
    'phase', 'i', 'ii', 'iii', 'iv', 'randomized', 'double-blind', 'placebo', 'controlled', 'open-label', 'multi-center',
    'active', 'control', 'placebo-controlled', 'subject', 'subjects', 'enrolled', 'eligibility', 'eligible',
    'criteria', 'inclusion', 'exclusion', 'male', 'female', 'healthy', 'history', 'prior', 'current', 'concomitant'
}


class ClusteringService:
    """
    Handles dynamic clustering of search results.
    """

    def __init__(self, bert_service: Any):
        """
        Initialize with BERTSearchService instance to fetch document embeddings.
        """
        self.bert_service = bert_service

    def cluster_search_results(
        self,
        results: List[Dict[str, Any]],
        n_clusters: int = 4
    ) -> Dict[str, Any]:
        """
        Clusters the search results and generates visualization data.

        Args:
            results: List of search result dictionaries containing "doc_id" and "full_text" or "text".
            n_clusters: Target number of clusters (will adjust down if results are fewer than clusters).

        Returns:
            Dictionary containing:
                - scatter_data: List of dicts with keys (x, y, cluster_id, cluster_label, doc_id, snippet)
                - cluster_labels: Dict mapping cluster_id to label string
                - grouped_results: Dict mapping cluster_id to list of result dicts
        """
        if not results:
            return {
                "scatter_data": [],
                "cluster_labels": {},
                "grouped_results": {}
            }

        # Adjust cluster count if results are fewer than requested clusters
        n_clusters = min(n_clusters, len(results))
        if n_clusters < 1:
            n_clusters = 1

        # 1. Fetch vectors for each result
        doc_ids = []
        vectors = []
        valid_results = []

        for r in results:
            doc_id = r["doc_id"]
            # Try to get pre-calculated SBERT vector
            vec = self.bert_service.get_vector(doc_id)

            # Fallback: encode text dynamically if vector isn't found
            if vec is None:
                text = r.get("full_text") or r.get("text") or ""
                if text:
                    vec = self.bert_service.model.encode(text)
                else:
                    # Zeros vector if no text exists
                    vec = np.zeros(self.bert_service.model.dim)

            doc_ids.append(doc_id)
            vectors.append(vec)
            valid_results.append(r)

        X = np.array(vectors, dtype=np.float32)

        # 2. Run KMeans Clustering
        # If there's only 1 document, assign all to cluster 0
        if len(results) == 1:
            labels = np.array([0])
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(X)

        # 3. Reduce dimensions to 2D using PCA
        # If less than 2 documents, pad coordinates with zeros
        if len(results) >= 2:
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X)
        else:
            coords = np.zeros((len(results), 2))

        # 4. Extract terms for each cluster to label it
        cluster_texts = collections.defaultdict(list)
        for i, label in enumerate(labels):
            doc_id = doc_ids[i]
            # Get full text for text analysis
            doc_data = None
            if hasattr(self.bert_service, "document_store") and self.bert_service.document_store:
                doc_data = self.bert_service.document_store.get_doc(doc_id)
            text = doc_data["text"] if doc_data else (valid_results[i].get("full_text") or valid_results[i].get("text") or "")
            cluster_texts[int(label)].append(text)


        cluster_labels = {}
        for cluster_id in range(n_clusters):
            texts = cluster_texts[cluster_id]
            label_text = self._generate_cluster_label(texts)
            cluster_labels[cluster_id] = f"Cluster {cluster_id}: {label_text}"

        # 5. Format scatter plot data
        scatter_data = []
        grouped_results = collections.defaultdict(list)

        for i, label in enumerate(labels):
            cluster_id = int(label)
            r = valid_results[i]
            
            # Short preview for plot hovers
            snippet = r.get("text", "")
            if len(snippet) > 150:
                snippet = snippet[:150] + "..."

            scatter_data.append({
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "cluster_id": cluster_id,
                "cluster_label": cluster_labels[cluster_id],
                "doc_id": doc_ids[i],
                "snippet": snippet,
                "score": r.get("score", 0.0)
            })

            # Add cluster info to result dict
            r_copy = r.copy()
            r_copy["cluster_id"] = cluster_id
            r_copy["cluster_label"] = cluster_labels[cluster_id]
            grouped_results[cluster_id].append(r_copy)

        return {
            "scatter_data": scatter_data,
            "cluster_labels": cluster_labels,
            "grouped_results": dict(grouped_results)
        }

    def _generate_cluster_label(self, texts: List[str]) -> str:
        """
        Helper method to extract the most representative words in a cluster.
        Filters out medical and common English stop words.
        """
        words = []
        for text in texts:
            # Tokenize words, lowercase and keep alphabetic characters
            tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            words.extend([t for t in tokens if t not in STOP_WORDS])

        if not words:
            return "General Trial Info"

        # Count frequencies
        counter = collections.Counter(words)
        # Take the top 3 most common keywords
        most_common = [word for word, count in counter.most_common(3)]
        
        # Capitalize for labels
        return ", ".join([w.upper() for w in most_common])
