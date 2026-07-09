"""
Global seeding utility for reproducible experiments.

Seeds Python's `random`, NumPy, and PyTorch (CPU + all CUDA devices).
Call once at the start of every experiment (and before any model
instantiation) so weight initialization, dropout, and data shuffling
are reproducible.

Note: full bit-wise determinism on GPU would additionally require
`torch.use_deterministic_algorithms(True)` and disabling cuDNN
benchmarking, which slows training. We deliberately keep those off;
seeding removes the dominant run-to-run variance (init + shuffling).
"""

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make hash-based orderings stable across runs as well
    os.environ["PYTHONHASHSEED"] = str(seed)
