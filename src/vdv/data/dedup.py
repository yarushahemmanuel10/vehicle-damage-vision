"""Near-duplicate detection with perceptual hashes (D-006)."""
import imagehash
import numpy as np


def phash_bits(img):
    """64-bit perceptual hash of a PIL image, as a flat array of 64 booleans."""
    return imagehash.phash(img).hash.flatten()


def hamming_matrix(a, b, chunk=256):
    """Hamming distances between every row of a (n x 64) and every row of b (m x 64) -> n x m integers."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    out = np.empty((len(a), len(b)), np.int16)
    for i in range(0, len(a), chunk):
        out[i:i + chunk] = (a[i:i + chunk, None, :] != b[None, :, :]).sum(axis=-1)
    return out


def removal_target(pair):
    """Which photo a duplicate pair removes (D-006 amendment): never a train photo.
    val/test vs train -> the val/test photo; val vs test -> the val photo; train vs train -> None."""
    splits = {pair["query_split"], pair["ref_split"]}
    if splits == {"train"}:
        return None
    if splits == {"val", "test"}:
        return ("val", pair["query"] if pair["query_split"] == "val" else pair["ref"])
    return (pair["query_split"], pair["query"]) if pair["query_split"] != "train" else (pair["ref_split"], pair["ref"])
