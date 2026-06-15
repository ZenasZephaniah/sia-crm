# src/utils.py
import os
import random
import numpy as np
import torch

class Config:
    SEED = 42
    MODEL_NAME = "microsoft/deberta-v3-small"
    NLI_MODEL_NAME = "valhalla/distilbart-mnli-12-3"  # Lightweight for fast, accurate zero-shot inference
    MAX_LEN = 64
    BATCH_SIZE = 16
    EPOCHS = 3
    LR = 2e-5
    
    # Priority mapping
    PRIORITY_MAP = {
        "Low": 1,
        "Medium": 2,
        "High": 3,
        "Critical": 4
    }
    
    # Reverse mapping for visualization
    REVERSE_PRIORITY_MAP = {v: k for k, v in PRIORITY_MAP.items()}

def seed_everything(seed=Config.SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False