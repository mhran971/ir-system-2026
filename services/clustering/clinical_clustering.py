# services/clustering/clinical_clustering.py
"""
Clinical Trials Clusterer Service.
Classifies the entire 241,006 documents of ClinicalTrials 2017 into medical topics
using domain-specific keyword matching, and stores/retrieves these assignments
from SQLite for sub-millisecond retrieval and global statistical overview.
"""

import sqlite3
import os
from typing import Dict, List, Any, Optional

class ClinicalTrialsClusterer:
    """
    Categorizes ClinicalTrials documents into hierarchical medical groups.
    """
    
    # 1. Cancer Type (🩺 Cancer Type)
    CANCER_TYPES = {
        'breast': [
            'breast cancer', 'breast carcinoma', 'mammary cancer', 'mammary carcinoma', 
            'breast neoplasm', 'breast tumor', 'ductal carcinoma', 'lobular carcinoma', 
            'her2-positive breast', 'triple-negative breast', 'ibc', 'tnbc', 'breast lesion'
        ],
        'lung': [
            'lung cancer', 'pulmonary carcinoma', 'non-small cell lung', 'nsclc', 
            'small cell lung', 'sclc', 'lung neoplasm', 'bronchial carcinoma', 
            'bronchogenic carcinoma', 'pulmonary adenocarcinoma', 'thoracic oncology', 'lung adenocarcinoma'
        ],
        'colorectal': [
            'colorectal cancer', 'colon cancer', 'rectal cancer', 'colorectal carcinoma', 
            'colon carcinoma', 'rectal carcinoma', 'colorectal neoplasm', 'colon neoplasm', 
            'rectal neoplasm', 'bowel cancer', 'rectosigmoid', 'colorectal adenocarcinoma'
        ],
        'cardiovascular': [
            'cardiovascular', 'heart disease', 'coronary', 'cardiac', 'myocardial', 
            'heart failure', 'stroke', 'arrhythmia', 'angina', 'ischemia', 'thrombosis', 
            'infarction', 'hypertensive', 'vascular disease', 'atrial fibrillation', 'cardiomyopathy'
        ],
        'neurological': [
            'neurological', 'brain cancer', 'meningioma', 'glioblastoma', 'alzheimer', 
            'parkinson', 'neuropathy', 'central nervous system', 'brain tumor', 'glioma', 
            'multiple sclerosis', 'epilepsy', 'dementia', 'neurodegenerative', 'cns tumor'
        ],
        'hematological': [
            'leukemia', 'leukaemia', 'lymphoma', 'myeloma', 'hematological', 'blood cancer', 
            'hodgkin', 'non-hodgkin', 'myelodysplastic', 'mds', 'cll', 'aml', 'all', 'cml', 
            'hematologic', 'myelofibrosis', 'thrombocytopenia', 'neutropenia', 'lymphocytic', 'myelogenous'
        ],
        'sarcoma': [
            'liposarcoma', 'sarcoma', 'soft tissue sarcoma', 'leiomyosarcoma', 'osteosarcoma', 
            'ewing', 'gist', 'gastrointestinal stromal tumor', 'rhabdomyosarcoma', 
            'synovial sarcoma', 'angiosarcoma', 'chondrosarcoma', 'fibrosarcoma', 'bone tumor'
        ],
        'pancreatic': [
            'pancreatic cancer', 'pancreatic carcinoma', 'pancreatic adenocarcinoma', 
            'pancreatic ductal', 'pancreatic neuroendocrine', 'pnet', 'pancreatic neoplasm'
        ],
        'melanoma': [
            'melanoma', 'skin cancer', 'skin carcinoma', 'cutaneous melanoma', 
            'uveal melanoma', 'melanocytic', 'skin neoplasm'
        ],
        'liver': [
            'liver cancer', 'hepatocellular', 'hcc', 'hepatic cancer', 'cholangiocarcinoma', 
            'hepatic neoplasm', 'hepatoma', 'liver metastasis', 'biliary tract'
        ]
    }

    # 2. Treatment Type (💊 Treatment Type)
    TREATMENT_TYPES = {
        'chemotherapy': [
            'chemotherapy', 'chemo', 'cytotoxic', 'doxorubicin', 'cisplatin', 'paclitaxel', 
            'docetaxel', 'fluorouracil', 'gemcitabine', 'carboplatin', 'oxaliplatin', 
            'temozolomide', 'irinotecan', 'etoposide', 'alkylating', 'antimetabolite'
        ],
        'immunotherapy': [
            'immunotherapy', 'immune', 'pd-1', 'pd-l1', 'pembrolizumab', 'nivolumab', 
            'car-t', 'checkpoint inhibitor', 'ipilimumab', 'atezolizumab', 'durvalumab', 
            'adoptive cell transfer', 'interleukin', 'interferon', 'vaccine therapy', 'chimeric antigen'
        ],
        'targeted': [
            'targeted therapy', 'inhibitor', 'monoclonal antibody', 'kinase inhibitor', 
            'tyrosine kinase', 'imatinib', 'trastuzumab', 'erlotinib', 'gefitinib', 
            'bevacizumab', 'cetuximab', 'rituximab', 'lapatinib', 'vemurafenib', 
            'dabrafenib', 'trametinib', 'crizotinib', 'alectinib', 'parp inhibitor', 'mator inhibitor'
        ],
        'radiotherapy': [
            'radiotherapy', 'radiation', 'x-ray', 'brachytherapy', 'irradiation', 
            'proton beam', 'radiosurgery', 'sbrt', 'imrt', 'external beam'
        ],
        'surgery': [
            'surgery', 'resection', 'operation', 'surgical', 'lumpectomy', 'mastectomy', 
            'lobectomy', 'excision', 'prostatectomy', 'colectomy', 'operative'
        ],
        'combination': [
            'combination therapy', 'combined therapy', 'combination of', 'chemoradiation', 
            'combo', 'concurrent', 'in combination with', 'multimodality'
        ],
        'gene_therapy': [
            'gene therapy', 'crispr', 'vector delivery', 'gene transfer', 'antisense', 
            'rnai', 'sirna', 'gene editing', 'gene correction', 'transgene'
        ]
    }

    # 3. Genes/Mutations (🧬 Genes/Mutations)
    GENES = {
        'kras': ['kras', 'k-ras', 'kras mutation', 'kras g12c', 'kras g12d'],
        'egfr': ['egfr', 'epidermal growth factor', 'erbb1', 'egfr mutation', 'egfr t790m'],
        'braf': ['braf', 'b-raf', 'braf mutation', 'braf v600e'],
        'her2': ['her2', 'erbb2', 'neu', 'her-2', 'her2 amplification'],
        'cdk4_6': ['cdk4', 'cdk6', 'cdk4/6', 'cyclin-dependent kinase 4', 'cyclin-dependent kinase 6', 'palbociclib', 'ribociclib', 'abemaciclib'],
        'brca': ['brca1', 'brca2', 'brca', 'brca mutation', 'brca1/2'],
        'alk': ['alk', 'anaplastic lymphoma kinase', 'alk rearrangement', 'alk fusion'],
        'tp53': ['tp53', 'p53', 'tp53 mutation', 'tumor protein p53']
    }

    # 4. Cancer Stage (🔬 Cancer Stage)
    STAGES = {
        'early': [
            'early stage', 'stage i', 'stage ii', 'localized', 'early-stage', 
            'non-metastatic', 'operable', 'resectable', 'stage 1', 'stage 2'
        ],
        'advanced': [
            'advanced stage', 'locally advanced', 'stage iii', 'advanced cancer', 
            'advanced solid', 'stage 3', 'unresectable', 'inoperable'
        ],
        'metastatic': [
            'metastatic', 'metastasis', 'stage iv', 'secondaries', 'disseminated', 
            'distant metastasis', 'metastases', 'stage 4', 'oligometastatic'
        ],
        'recurrent': [
            'recurrent', 'relapsed', 'recurrence', 'refractory', 'progression', 
            'progressive', 'resistant', 'second-line', 'salvage'
        ]
    }

    # 5. Demographics (🧒 Demographics)
    DEMOGRAPHICS = {
        'pediatric': [
            'pediatric', 'pediatrics', 'children', 'child', 'childhood', 'adolescent', 
            'infant', 'newborn', 'neonatal', 'toddler', 'young patient'
        ],
        'adult': [
            'adult', 'adults', 'age >= 18', '18 years to', '18 and older', 'adult population'
        ],
        'elderly': [
            'elderly', 'older adults', 'aged 65', 'geriatric', 'senior', 'older patients', 'aged >= 65'
        ],
        'female': [
            'female', 'women', 'woman', 'girls', 'maternal', 'pregnancy', 'breastfeeding'
        ],
        'male': [
            'male', 'men', 'man', 'boys', 'paternal', 'prostate'
        ]
    }

    # 6. Comorbidities (❤️ Comorbidities)
    COMORBIDITIES = {
        'hypertension': ['hypertension', 'high blood pressure', 'bp control'],
        'diabetes': ['diabetes', 'diabetic', 'hyperglycemia', 'type 2 diabetes', 't2dm', 'insulin-dependent'],
        'copd': ['copd', 'chronic obstructive pulmonary', 'emphysema', 'bronchitis', 'asthma', 'pulmonary disease'],
        'heart_failure': ['heart failure', 'congestive heart failure', 'chf', 'ejection fraction', 'lvef'],
        'depression': ['depression', 'depressive', 'anxiety', 'psychiatric', 'mental health', 'bipolar']
    }

    CATEGORY_MAPPING = {
        'cancer_type': CANCER_TYPES,
        'treatment_type': TREATMENT_TYPES,
        'gene': GENES,
        'stage': STAGES,
        'demographics': DEMOGRAPHICS,
        'comorbidity': COMORBIDITIES
    }

    # Human readable labels for the UI
    CATEGORY_LABELS = {
        'cancer_type': '🩺 Cancer Type',
        'treatment_type': '💊 Treatment Type',
        'gene': '🧬 Genes/Mutations',
        'stage': '🔬 Cancer Stage',
        'demographics': '🧒 Demographics',
        'comorbidity': '❤️ Comorbidities'
    }

    def __init__(self, db_path: str = 'data/processed/doc_store.db'):
        self.db_path = db_path
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        """Creates the document_topics table if it does not exist."""
        if not os.path.exists(self.db_path):
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_topics (
                    doc_id TEXT PRIMARY KEY,
                    cancer_type TEXT,
                    treatment_type TEXT,
                    gene TEXT,
                    stage TEXT,
                    demographics TEXT,
                    comorbidity TEXT
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_topics_cancer ON document_topics (cancer_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_topics_treatment ON document_topics (treatment_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_topics_gene ON document_topics (gene)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_topics_stage ON document_topics (stage)")
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"⚠️ [ClinicalTrialsClusterer] Error initializing table: {e}")

    def classify_text(self, text: str) -> Dict[str, str]:
        """
        Classifies a document's text into each of the 6 medical topic categories.
        """
        text_lower = text.lower()
        assignments = {}

        for category_name, keyword_dict in self.CATEGORY_MAPPING.items():
            assigned = False
            for topic, keywords in keyword_dict.items():
                if any(kw in text_lower for kw in keywords):
                    assignments[category_name] = topic
                    assigned = True
                    break
            if not assigned:
                assignments[category_name] = 'other'

        return assignments

    def precompute_all_documents(self, force: bool = False):
        """
        Loads all documents from `documents` table, classifies them,
        and saves the classifications to `document_topics` table.
        """
        if not os.path.exists(self.db_path):
            print(f"❌ Database not found at {self.db_path}")
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if already computed
        if not force:
            cursor.execute("SELECT COUNT(*) FROM document_topics")
            count = cursor.fetchone()[0]
            if count > 0:
                print(f"ℹ️ [ClinicalTrialsClusterer] Topic mappings already exist ({count} docs). Skipping precomputation.")
                conn.close()
                return

        print("🚀 [ClinicalTrialsClusterer] Precomputing topics for the entire dataset...")
        
        # Clear existing mappings
        cursor.execute("DELETE FROM document_topics")
        conn.commit()

        # Load in batches to prevent high memory usage
        batch_size = 10000
        offset = 0
        total_inserted = 0

        while True:
            cursor.execute("SELECT doc_id, text FROM documents LIMIT ? OFFSET ?", (batch_size, offset))
            rows = cursor.fetchall()
            if not rows:
                break

            insert_batch = []
            for doc_id, text in rows:
                text = text or ""
                topics = self.classify_text(text)
                insert_batch.append((
                    doc_id,
                    topics['cancer_type'],
                    topics['treatment_type'],
                    topics['gene'],
                    topics['stage'],
                    topics['demographics'],
                    topics['comorbidity']
                ))

            cursor.executemany("""
                INSERT INTO document_topics (
                    doc_id, cancer_type, treatment_type, gene, stage, demographics, comorbidity
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, insert_batch)
            conn.commit()
            
            total_inserted += len(rows)
            print(f"   Processed {total_inserted} documents...")
            offset += batch_size

        conn.close()
        print(f"✅ [ClinicalTrialsClusterer] Precomputed topic mappings for {total_inserted} documents successfully!")

    def get_global_counts(self) -> Dict[str, Dict[str, int]]:
        """
        Retrieves global statistics of document counts per category and topic.
        """
        counts = {}
        if not os.path.exists(self.db_path):
            return counts

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for category in self.CATEGORY_MAPPING.keys():
                cursor.execute(f"SELECT {category}, COUNT(*) FROM document_topics GROUP BY {category}")
                rows = cursor.fetchall()
                counts[category] = {row[0]: row[1] for row in rows}

            conn.close()
        except Exception as e:
            print(f"⚠️ [ClinicalTrialsClusterer] Error fetching global counts: {e}")

        return counts

    def get_document_topics(self, doc_ids: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Looks up topic assignments for a list of document IDs.
        """
        results = {}
        if not doc_ids or not os.path.exists(self.db_path):
            return results

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Batch query
            placeholders = ",".join(["?"] * len(doc_ids))
            query = f"SELECT doc_id, cancer_type, treatment_type, gene, stage, demographics, comorbidity FROM document_topics WHERE doc_id IN ({placeholders})"
            cursor.execute(query, doc_ids)
            rows = cursor.fetchall()

            for row in rows:
                results[row[0]] = {
                    'cancer_type': row[1],
                    'treatment_type': row[2],
                    'gene': row[3],
                    'stage': row[4],
                    'demographics': row[5],
                    'comorbidity': row[6]
                }

            conn.close()
        except Exception as e:
            print(f"⚠️ [ClinicalTrialsClusterer] Error fetching document topics: {e}")

        return results
