# tests/test_clustering.py
"""
Unit tests for ClusteringService.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from services.clustering.clustering_service import ClusteringService
from services.ranking.embeddings.bert_search_service import BERTSearchService


class MockEmbeddingModel:
    def __init__(self):
        self.dim = 384
        self.model_name = "MockModel"

    def encode(self, text):
        return np.random.randn(self.dim)


class MockBERTSearchService:
    def __init__(self):
        self.model = MockEmbeddingModel()
        self.document_store = None

    def get_vector(self, doc_id):
        # Return random normalized vector
        vec = np.random.randn(384)
        return vec / np.linalg.norm(vec)


def test_clustering_with_mock_service():
    mock_bert = MockBERTSearchService()
    service = ClusteringService(bert_service=mock_bert)

    # Mock results list
    results = [
        {"doc_id": "doc1", "text": "This is a trial about colon cancer treatments and chemotherapy.", "score": 0.95},
        {"doc_id": "doc2", "text": "A clinical study on breast cancer drug efficacy in female patients.", "score": 0.85},
        {"doc_id": "doc3", "text": "Research into brain tumors and surgical resection techniques.", "score": 0.75},
        {"doc_id": "doc4", "text": "Immunotherapy outcomes in advanced stage lung cancer patients.", "score": 0.65},
    ]

    # Cluster results into 2 clusters
    output = service.cluster_search_results(results, n_clusters=2)

    assert "scatter_data" in output
    assert "cluster_labels" in output
    assert "grouped_results" in output

    assert len(output["scatter_data"]) == 4
    assert len(output["cluster_labels"]) == 2
    assert len(output["grouped_results"]) == 2

    # Check structure of scatter data
    first_point = output["scatter_data"][0]
    assert "x" in first_point
    assert "y" in first_point
    assert "cluster_id" in first_point
    assert "cluster_label" in first_point
    assert "doc_id" in first_point
    assert "snippet" in first_point
    assert "score" in first_point

    # Ensure labels are capitalized keywords
    for cluster_id, label in output["cluster_labels"].items():
        assert f"Cluster {cluster_id}:" in label


if __name__ == "__main__":
    test_clustering_with_mock_service()
    print("✅ test_clustering_with_mock_service PASSED")
