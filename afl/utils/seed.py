"""One seed, set once, at the top of every entry point.

Anything that draws randomness without going through here is a reproducibility bug.
"""

from __future__ import annotations

import os
import random

DEFAULT_SEED = 1337


def set_all_seeds(seed: int = DEFAULT_SEED) -> int:
    """Seed python, numpy, and (if installed) torch. Returns the seed for logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy is a hard dep in practice; smoke tests may run without it
        pass

    try:  # optional — only present with the `deep` extra
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass

    return seed


def child_seed(seed: int, *parts: object) -> int:
    """Derive a stable sub-seed (per round, per vector) without global reseeding."""
    import zlib

    key = "|".join(str(p) for p in parts).encode()
    return (seed * 1_000_003 + zlib.crc32(key)) % (2**31 - 1)


def rng(seed: int):
    """A local numpy Generator — preferred over touching the global numpy state."""
    import numpy as np

    return np.random.default_rng(seed)
