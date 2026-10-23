"""Classification crops: D-014 margin, D-015 minimum window, D-016 aspect cap.

The crop window is computed in the ORIGINAL photo, in whole pixels:
  1) box + MARGIN (share of box width/height on each side), clipped to the image
  2) any side shorter than MIN_SIDE grown to exactly MIN_SIDE around its centre
  3) the shorter side grown until long / short <= MAX_ASPECT
Steps 2 and 3 use real image context and shift the window to stay inside the image.
Stored crops keep their aspect ratio with the long side resized to STORE_LONG_SIDE (unpadded);
padding to a square happens at load time, after augmentation.
"""
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

CROP_VERSION = "1"            # bump by hand when the crop rules change; the code hash changes automatically
MARGIN = 0.10                 # D-014
MIN_SIDE = 64                 # D-015
MAX_ASPECT = 3.0              # D-016
STORE_LONG_SIDE = 224
PAD_VALUE = 128               # mid-grey

REPO_ROOT = Path(__file__).resolve().parents[3]       # src/vdv/data/crops.py -> repo root (editable install)
VERSIONED_FILES = [Path(__file__).resolve(),          # the crop rules
                   REPO_ROOT / "scripts" / "build_crops.py"]   # the build script (skipped if missing)


def hash_files(paths):
    """Short SHA-1 over the names and contents of the files that exist, in order. Missing files are skipped."""
    h = hashlib.sha1()
    for p in map(Path, paths):
        if p.exists():
            h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()[:8]


def crop_code_version():
    """Version tied to the exact contents of the crop rules and the build script, e.g. 'v1-3fa9c2d1'."""
    return f"v{CROP_VERSION}-{hash_files(VERSIONED_FILES)}"


def _grow(lo, hi, target, limit):
    """Grow the whole-pixel interval [lo, hi) to length `target` around its centre, shifted to stay
    inside [0, limit). Never shrinks; if target > limit the result is the full range [0, limit)."""
    target = min(int(target), limit)
    if hi - lo >= target:
        return lo, hi
    lo -= (target - (hi - lo)) // 2
    lo = max(0, min(lo, limit - target))
    return lo, lo + target


def crop_window(box, img_w, img_h, margin=MARGIN, min_side=MIN_SIDE, max_aspect=MAX_ASPECT):
    """Final crop window (x0, y0, x1, y1) in whole pixels for a COCO box [x, y, w, h], plus flags."""
    x, y, w, h = box
    x0, y0 = max(0, math.floor(x - margin * w)), max(0, math.floor(y - margin * h))
    x1, y1 = min(img_w, math.ceil(x + w + margin * w)), min(img_h, math.ceil(y + h + margin * h))

    nx0, nx1 = _grow(x0, x1, min_side, img_w)
    ny0, ny1 = _grow(y0, y1, min_side, img_h)
    flags = {"min_window_applied": (nx0, nx1, ny0, ny1) != (x0, x1, y0, y1),
             "aspect_capped": False, "aspect_cap_incomplete": False}
    x0, x1, y0, y1 = nx0, nx1, ny0, ny1

    W, H = x1 - x0, y1 - y0
    if W > max_aspect * H:
        y0, y1 = _grow(y0, y1, math.ceil(W / max_aspect), img_h)
        flags["aspect_capped"] = True
    elif H > max_aspect * W:
        x0, x1 = _grow(x0, x1, math.ceil(H / max_aspect), img_w)
        flags["aspect_capped"] = True
    W, H = x1 - x0, y1 - y0
    flags["aspect_cap_incomplete"] = max(W, H) > max_aspect * min(W, H)
    return (x0, y0, x1, y1), flags


def resize_long_side(crop, long_side=STORE_LONG_SIDE):
    """Resize so the longer side is `long_side`, keeping the aspect ratio."""
    h, w = crop.shape[:2]
    s = long_side / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    return cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)


def pad_to_square(img, size=STORE_LONG_SIDE, value=PAD_VALUE):
    """Centre an image (both sides <= size) on a size x size canvas of `value`."""
    h, w = img.shape[:2]
    if h > size or w > size:
        raise ValueError(f"image {w}x{h} larger than {size}x{size}")
    out = np.full((size, size) + img.shape[2:], value, dtype=img.dtype)
    y0, x0 = (size - h) // 2, (size - w) // 2
    out[y0:y0 + h, x0:x0 + w] = img
    return out


def check_crop_cache(crop_dir):
    """Refuse a crop folder built by different crop code. Returns its meta.json contents."""
    meta = json.loads((Path(crop_dir) / "meta.json").read_text())
    if meta.get("version") != crop_code_version():
        raise RuntimeError(f"Crop cache {crop_dir} is version {meta.get('version')}, but the code is "
                           f"{crop_code_version()}: rebuild with scripts/build_crops.py")
    return meta
