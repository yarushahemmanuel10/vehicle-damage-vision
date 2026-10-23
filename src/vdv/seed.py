"""Fix all random number generators so that runs can be repeated."""
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU and GPU).

    deterministic=True also makes cuDNN use the same algorithms every run.
    This reduces run-to-run variation on the GPU (it does not guarantee
    identical results) and makes training slower.
    """
    random.seed(seed)                       # Python's own random module
    np.random.seed(seed)                    # NumPy (augmentations, sampling)
    torch.manual_seed(seed)                 # PyTorch on CPU (weight init, shuffling)
    torch.cuda.manual_seed_all(seed)        # PyTorch on every GPU
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
