import os

class Config:
    DATA_PATH = "data/raw"
    PROCESSED_PATH = "data/processed"
    INDEX_PATH = "data/index"
    DATASET_NAME = "msmarco-passage/dev/small" 
    MAX_DOCS = 50000  # للاختبار السريع
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    RRF_K = 60