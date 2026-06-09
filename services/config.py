class Config:
    DATA_PATH = "data/raw"
    PROCESSED_PATH = "data/processed"
    INDEX_PATH = "data/index"
    DATASET_NAME = "msmarco-passage/train"
    MAX_DOCS = 200000
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    RRF_K = 60
