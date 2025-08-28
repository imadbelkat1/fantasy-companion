from pathlib import Path
from typing import Dict, Any
import json


class Config:
    """Configuration management for FDR service"""

    # Data paths
    DATA_PATH = Path("../Data-Service/data/fbref_data.json")
    OUTPUT_DIR = Path("output")
    MODELS_DIR = Path("models")

    # Supported leagues
    LEAGUES = ["epl", "laliga"]

    # Model parameters
    MODEL_CONFIG = {
        "ridge_alpha": 1.0,
        "lgb_params": {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "max_depth": 6,
            "learning_rate": 0.1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1
        }
    }

    # Rating parameters
    RATING_CONFIG = {
        "base_rating": 50,
        "scale_factor": 12,
        "min_rating": 0,
        "max_rating": 100,
        "shrinkage_lambda": 0.3
    }

    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        cls.MODELS_DIR.mkdir(exist_ok=True)