import numpy as np
from PIL import Image

from vdv.data.dedup import hamming_matrix, phash_bits, removal_target


def test_hamming_matrix_counts_differing_bits():
    zeros, ones = np.zeros(64, bool), np.ones(64, bool)
    half = np.array([True] * 32 + [False] * 32)
    d = hamming_matrix([zeros, ones], [zeros, half])
    assert d.tolist() == [[0, 32], [64, 32]]


def test_same_image_has_distance_zero():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8))
    a, b = phash_bits(img), phash_bits(img.copy())
    assert a.shape == (64,) and hamming_matrix([a], [b])[0, 0] == 0


def pair(qs, q, rs, r):
    return {"query_split": qs, "query": q, "ref_split": rs, "ref": r}


def test_removal_never_touches_train():
    assert removal_target(pair("val", "v.jpg", "train", "t.jpg")) == ("val", "v.jpg")
    assert removal_target(pair("test", "x.jpg", "train", "t.jpg")) == ("test", "x.jpg")
    assert removal_target(pair("train", "t.jpg", "test", "x.jpg")) == ("test", "x.jpg")
    assert removal_target(pair("train", "a.jpg", "train", "b.jpg")) is None


def test_val_test_duplicate_removes_val():
    assert removal_target(pair("val", "v.jpg", "test", "x.jpg")) == ("val", "v.jpg")
    assert removal_target(pair("test", "x.jpg", "val", "v.jpg")) == ("val", "v.jpg")
