"""Helpers for COCO-style boxes [x, y, width, height]."""
import numpy as np

SMALL_MAX_AREA = 32 ** 2       # COCO convention: small < 32², medium < 96², large >= 96²
MEDIUM_MAX_AREA = 96 ** 2


def size_bucket(area):
    """COCO size bucket for an area (number or array): 'small', 'medium' or 'large'."""
    area = np.asarray(area, dtype=float)
    out = np.select([area < SMALL_MAX_AREA, area < MEDIUM_MAX_AREA], ["small", "medium"], "large")
    return out.item() if out.ndim == 0 else out


def intersection_area(a, b):
    """Overlap area in pixels² of two [x, y, w, h] boxes; 0.0 if they do not overlap."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    return ix * iy
