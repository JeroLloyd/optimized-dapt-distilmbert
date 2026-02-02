"""Utility helpers for thesis scripts: reproducibility, logging, and IO helpers.

Keep helpers minimal and dependency-free; suitable for artifact evaluation.
"""
import logging
import os
import json
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False):
    """Set all relevant random seeds for reproducibility.

    deterministic: if True, enable deterministic CuDNN behavior (may slow training).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(name: str = __name__, level: int = logging.INFO):
    """Create a simple logger used across scripts."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, indent=2)


def read_text_lines(path: str, max_lines: int = None):
    out = []
    with open(path, 'r', encoding='utf-8') as fh:
        for i, l in enumerate(fh):
            if max_lines is not None and i >= max_lines:
                break
            out.append(l.rstrip('\n'))
    return out
