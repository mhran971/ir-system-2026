class Config:
    DATA_PATH = "data/raw"
    PROCESSED_PATH = "data/processed"
    INDEX_PATH = "data/index"
    DATASET_NAME = "clinicaltrials/2017/trec-pm-2017"
    MAX_DOCS = None
    
    # ✅ استخدام النموذج الطبي كافتراضي
    EMBEDDING_MODEL = "clinical"  # أو "pritamdeka/S-PubMedBert-MS-MARCO"
    
    RRF_K = 60