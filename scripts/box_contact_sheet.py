"""Contact sheets for the A2 crop decisions (TRAIN split only, D-013).

Usage:
    python scripts/box_contact_sheet.py --root /content/data/CarDD_release/CarDD_COCO
Writes two PNGs to outputs/figures_private/ (git-ignored: they contain dataset images).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from box_stats import load_train_boxes      # same folder (scripts/)

SIZE = 224
PAD_VALUE = 128                              # mid-grey for padding
LABEL_H = 34
MARGINS = [0.0, 0.10, 0.25]
MIN_WINDOW = 64                              # minimum-window preview for tiny boxes


# ---------- crop and resize methods (previews; A2 will implement the chosen one in the package) ----------
def resize_to(img, w, h):
    shrinking = w < img.shape[1] and h < img.shape[0]
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)


def crop_margin(img, box, margin):
    """Box plus margin (share of box width/height on each side), clipped to the image."""
    H, W = img.shape[:2]
    x, y, w, h = box
    x0, y0 = max(0, int(np.floor(x - margin * w))), max(0, int(np.floor(y - margin * h)))
    x1, y1 = min(W, int(np.ceil(x + w + margin * w))), min(H, int(np.ceil(y + h + margin * h)))
    return img[y0:y1, x0:x1]


def min_window_crop(img, box, min_side=MIN_WINDOW):
    """Grow each side of the box to at least min_side px around its centre (real image context),
    shifted to stay inside the image. Sides already >= min_side are unchanged."""
    H, W = img.shape[:2]
    x, y, w, h = box
    nw, nh = max(w, min_side), max(h, min_side)
    cx, cy = x + w / 2, y + h / 2
    x0 = min(max(cx - nw / 2, 0), max(W - nw, 0))
    y0 = min(max(cy - nh / 2, 0), max(H - nh, 0))
    x0i, y0i = int(np.floor(x0)), int(np.floor(y0))
    x1i, y1i = min(W, int(np.ceil(x0 + nw))), min(H, int(np.ceil(y0 + nh)))
    return img[y0i:y1i, x0i:x1i]


def stretch(crop):
    return resize_to(crop, SIZE, SIZE)


def pad(crop):
    h, w = crop.shape[:2]
    s = SIZE / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    canvas = np.full((SIZE, SIZE, 3), PAD_VALUE, np.uint8)
    y0, x0 = (SIZE - nh) // 2, (SIZE - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resize_to(crop, nw, nh)
    return canvas


def square_expand(img, box, margin=0.0):
    """Square region around the box centre in the original photo; shifted to stay inside the
    image, padded only if the square is larger than the image."""
    H, W = img.shape[:2]
    x, y, w, h = box
    side = max(w, h) * (1 + 2 * margin)
    cx, cy = x + w / 2, y + h / 2

    def start(c, limit):
        s0 = c - side / 2
        return min(max(s0, 0), limit - side) if side <= limit else (limit - side) / 2

    x0, y0, s = int(round(start(cx, W))), int(round(start(cy, H))), max(1, int(round(side)))
    canvas = np.full((s, s, 3), PAD_VALUE, np.uint8)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + s, W), min(y0 + s, H)
    canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return resize_to(canvas, SIZE, SIZE)


# ---------- layout ----------
def tile(img224, lines):
    strip = np.full((LABEL_H, SIZE, 3), 255, np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(strip, t, (4, 13 + 14 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([img224, strip])


def title(text, width):
    strip = np.full((30, width, 3), 235, np.uint8)
    cv2.putText(strip, text, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return strip


def grid(sections, n_cols):
    """sections: list of (title, rows); each row is a list of tiles."""
    blank = np.full((SIZE + LABEL_H, SIZE, 3), 255, np.uint8)
    width = n_cols * SIZE
    parts = []
    for name, rows in sections:
        parts.append(title(name, width))
        for row in rows:
            row = row + [blank] * (n_cols - len(row))
            parts.append(np.hstack(row))
    return np.vstack(parts)


def label(r):
    return [f"{r.cls}  #{r.ann_id}", f"{r.w:.0f}x{r.h:.0f}px  aspect {r.aspect:.1f}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="Path to CarDD_COCO")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "outputs/figures_private")
    args = ap.parse_args()

    coco = json.loads((args.root / "annotations" / "instances_train2017.json").read_text())
    file_of = {im["id"]: im["file_name"] for im in coco["images"]}
    cache = {}

    def image(image_id):
        if image_id not in cache:
            cache[image_id] = cv2.imread(str(args.root / "train2017" / file_of[image_id]))
        return cache[image_id]

    df = load_train_boxes(args.root)
    order = df["cls"].value_counts().index.tolist()
    box = lambda r: (r.x, r.y, r.w, r.h)

    # --- Sheet 1: extremes per class (padded, no margin) ---
    sections = []
    for c in order:
        d = df[df["cls"] == c]
        small = d.sort_values(["short_side", "ann_id"]).head(5)
        elong = d.sort_values(["aspect", "ann_id"], ascending=[False, True]).head(5)
        sections.append((f"{c}: 5 smallest (shorter side) | padded, no margin",
                         [[tile(pad(crop_margin(image(r.image_id), box(r), 0.0)), label(r))
                           for r in small.itertuples(index=False)]]))
        sections.append((f"{c}: 5 most elongated | padded, no margin",
                         [[tile(pad(crop_margin(image(r.image_id), box(r), 0.0)), label(r))
                           for r in elong.itertuples(index=False)]]))
    args.out.mkdir(parents=True, exist_ok=True)
    p1 = args.out / "contact_extremes_train.png"
    cv2.imwrite(str(p1), grid(sections, 5))

    # --- Sheet 2: resize methods, margins, minimum window ---
    typical = []
    for c in order:
        d = df[df["cls"] == c].copy()
        d["dist"] = (d["short_side"] - d["short_side"].median()).abs()
        typical.append(d.sort_values(["dist", "ann_id"]).iloc[0])
    elongated = []
    for c in ["scratch", "crack"]:
        d = df[(df["cls"] == c) & (df["short_side"] >= 16)]
        elongated.append(d.sort_values(["aspect", "ann_id"], ascending=[False, True]).iloc[0])
    smallest_cracks = df[df["cls"] == "crack"].sort_values(["short_side", "ann_id"]).head(5)

    method_rows = []
    for r in typical + elongated:
        img, b = image(r.image_id), box(r)
        crop = crop_margin(img, b, 0.0)
        method_rows.append([tile(stretch(crop), label(r)[:1] + ["stretch"]),
                            tile(pad(crop), label(r)[:1] + ["pad"]),
                            tile(square_expand(img, b, 0.0), label(r)[:1] + ["square-expand"])])
    margin_rows = []
    for r in typical:
        img, b = image(r.image_id), box(r)
        margin_rows.append([tile(pad(crop_margin(img, b, m)), label(r)[:1] + [f"margin {int(m * 100)}% (padded)"])
                            for m in MARGINS])
    window_rows = []
    for r in smallest_cracks.itertuples(index=False):
        img, b = image(r.image_id), box(r)
        win = min_window_crop(img, b)
        window_rows.append([tile(pad(crop_margin(img, b, 0.0)), [label(r)[1], "plain, padded"]),
                            tile(pad(win), [f"window {win.shape[1]}x{win.shape[0]}px",
                                            f"min {MIN_WINDOW}px window, padded"])])
    p2 = args.out / "contact_methods_margins_train.png"
    cv2.imwrite(str(p2), grid([
        ("Resize methods at 224x224, no margin: stretch | pad | square-expand", method_rows),
        ("Margins 0% | 10% | 25% (share of box size on each side), padded", margin_rows),
        (f"5 smallest cracks: plain padded | min {MIN_WINDOW}px window (real context), padded", window_rows),
    ], 3))

    print("typical boxes (closest to class median shorter side):",
          {r.cls: int(r.ann_id) for r in typical})
    print("elongated boxes:", {r.cls: int(r.ann_id) for r in elongated})
    print("smallest cracks:", smallest_cracks["ann_id"].astype(int).tolist())
    print("written:", p1, "|", p2)


if __name__ == "__main__":
    main()
