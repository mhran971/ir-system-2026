# services/ranking/config.py
"""
Configuration constants for VSM TF-IDF system.
"""

from typing import List


class VSMTFIDFConfig:
    """Configuration settings for VSM TF-IDF."""
    
    # Search settings
    DEFAULT_TOP_K: int = 10
    TEXT_PREVIEW_LENGTH: int = 500
    
    # Data paths (in order of preference)
    DATA_PATHS: List[str] = [
        'data/processed/processed_docs_200000.pkl',
        'data/processed/processed_docs_5000.pkl',
        'data/processed/processed_docs.pkl'
    ]
    
    # IDF smoothing
    IDF_SMOOTHING: bool = True
    IDF_ADD_ONE_SMOOTHING: bool = True  # Add +1 to numerator and denominator
    
    # TF normalization
    TF_LENGTH_NORMALIZATION: bool = True
    
    # Preprocessing settings
    USE_STEMMING: bool = False
    USE_LEMMATIZATION: bool = False
    MIN_TOKEN_LENGTH: int = 2
    
    @classmethod
    def to_dict(cls) -> dict:
        """Convert config to dictionary."""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }