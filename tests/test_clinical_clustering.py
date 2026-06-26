# tests/test_clinical_clustering.py
"""
Unit tests for ClinicalTrialsClusterer.
"""

import sys
import os
import sqlite3
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from services.clustering.clinical_clustering import ClinicalTrialsClusterer

def test_classification():
    # Instantiate with a temporary SQLite database
    db_path = "tests/test_doc_store.db"
    
    # Clean up test db if exists
    if os.path.exists(db_path):
        os.remove(db_path)
        
    clusterer = ClinicalTrialsClusterer(db_path=db_path)
    
    # Test breast cancer and chemotherapy matching
    text_1 = "This clinical study targets breast cancer patients undergoing chemotherapy treatment."
    classes_1 = clusterer.classify_text(text_1)
    
    assert classes_1['cancer_type'] == 'breast'
    assert classes_1['treatment_type'] == 'chemotherapy'
    assert classes_1['gene'] == 'other'
    
    # Test lung cancer, targeted therapy and EGFR mutations
    text_2 = "A trial for NSCLC (non-small cell lung cancer) containing EGFR mutations using targeted tyrosine kinase inhibitors."
    classes_2 = clusterer.classify_text(text_2)
    
    assert classes_2['cancer_type'] == 'lung'
    assert classes_2['treatment_type'] == 'targeted'
    assert classes_2['gene'] == 'egfr'
    
    # Test sarcoma, CDK4 amplification and surgery
    text_3 = "Surgical resection of liposarcoma tumors with CDK4 amplifications."
    classes_3 = clusterer.classify_text(text_3)
    
    assert classes_3['cancer_type'] == 'sarcoma'
    assert classes_3['treatment_type'] == 'surgery'
    assert classes_3['gene'] == 'cdk4_6'
    
    # Clean up
    if os.path.exists(db_path):
        os.remove(db_path)
        
    print("✅ test_classification PASSED")

if __name__ == "__main__":
    test_classification()
