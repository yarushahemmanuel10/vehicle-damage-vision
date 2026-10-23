import random

import numpy as np
import torch

from vdv.seed import set_seed


def draw():
    """One number from each random generator we seed."""
    return random.random(), float(np.random.rand()), torch.rand(1).item()


def test_same_seed_gives_same_numbers():
    set_seed(123); first = draw()
    set_seed(123); second = draw()
    assert first == second


def test_different_seeds_give_different_numbers():
    set_seed(1); a = draw()
    set_seed(2); b = draw()
    assert a != b
