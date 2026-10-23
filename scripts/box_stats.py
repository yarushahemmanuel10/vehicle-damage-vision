"""Box statistics per class on the TRAIN split only (D-013), as evidence for the A2 crop decisions.

Usage:
    python scripts/box_stats.py --root /content/data/CarDD_release/CarDD_COCO
Writes outputs/metrics/box_stats_train.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vdv.data.boxes import intersection_area, size_bucket

PCTS = [5, 25, 50, 75, 95]
SHORT_SIDE_THRESHOLDS = [16, 32, 48, 64]
COVER_THRESHOLDS = [0.25, 0.50]
BUCKETS = ["small", "medium", "large"]


def load_train_boxes(root):
    coco = json.loads((root / "annotations" / "instances_train2017.json").read_text())
    cat = {c["id"]: c["name"] for c in coco["categories"]}
    img = {im["id"]: im for im in coco["images"]}
    rows = []
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        im = img[a["image_id"]]
        rows.append({"ann_id": a["id"], "image_id": a["image_id"], "cls": cat[a["category_id"]],
                     "x": x, "y": y, "w": w, "h": h,
                     "short_side": min(w, h), "aspect": max(w, h) / min(w, h),
                     "box_area": w * h, "mask_area": a["area"],
                     "w_frac": w / im["width"], "h_frac": h / im["height"]})
    df = pd.DataFrame(rows)
    df["size_by_box"] = size_bucket(df["box_area"].to_numpy())
    df["size_by_mask"] = size_bucket(df["mask_area"].to_numpy())   # pycocotools uses the annotation 'area'
    return df


def max_cover_other_class(df):
    """Per box: the largest share of its area covered by one box of a DIFFERENT class in the same image."""
    cover = {}
    for _, group in df.groupby("image_id"):
        boxes = list(group.itertuples(index=False))
        for a in boxes:
            best = 0.0
            for b in boxes:
                if b.ann_id != a.ann_id and b.cls != a.cls:
                    share = intersection_area((a.x, a.y, a.w, a.h), (b.x, b.y, b.w, b.h)) / a.box_area
                    best = max(best, share)
            cover[a.ann_id] = best
    return df["ann_id"].map(cover)


def spread(s, digits=1):
    out = {"min": round(float(s.min()), digits)}
    out.update({f"p{p}": round(float(np.percentile(s, p)), digits) for p in PCTS})
    out["max"] = round(float(s.max()), digits)
    return out


def describe(d):
    return {
        "count": int(len(d)),
        "w_px": spread(d["w"]),
        "h_px": spread(d["h"]),
        "short_side_px": spread(d["short_side"]),
        "aspect_long_over_short": {**{f"p{p}": round(float(np.percentile(d["aspect"], p)), 2) for p in [50, 75, 95]},
                                   "max": round(float(d["aspect"].max()), 2)},
        "median_w_frac_of_image": round(float(d["w_frac"].median()), 3),
        "median_h_frac_of_image": round(float(d["h_frac"].median()), 3),
        "size_by_box": {b: int((d["size_by_box"] == b).sum()) for b in BUCKETS},
        "size_by_mask": {b: int((d["size_by_mask"] == b).sum()) for b in BUCKETS},
        "short_side_below_px": {str(t): int((d["short_side"] < t).sum()) for t in SHORT_SIDE_THRESHOLDS},
        "covered_by_other_class": {"any": int((d["cover"] > 0).sum()),
                                   **{f"ge_{int(t * 100)}pct": int((d["cover"] >= t).sum()) for t in COVER_THRESHOLDS}},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="Path to CarDD_COCO")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "outputs/metrics/box_stats_train.json")
    args = ap.parse_args()

    df = load_train_boxes(args.root)
    df["cover"] = max_cover_other_class(df)
    order = df["cls"].value_counts().index.tolist()
    stats = {"split": "train", "per_class": {c: describe(df[df["cls"] == c]) for c in order},
             "all": describe(df),
             "settings": {"percentiles": PCTS, "short_side_thresholds_px": SHORT_SIDE_THRESHOLDS,
                          "cover_thresholds": COVER_THRESHOLDS,
                          "size_buckets": "COCO: small < 32^2, medium < 96^2, large >= 96^2"}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, indent=2))

    # --- Human-readable tables (ALL = every class together) ---
    pd.set_option("display.width", 220)
    table = {**stats["per_class"], "ALL": stats["all"]}
    for key in ["w_px", "h_px", "short_side_px", "aspect_long_over_short", "size_by_box", "size_by_mask",
                "short_side_below_px", "covered_by_other_class"]:
        print(f"\n=== {key} (train) ===")
        print(pd.DataFrame({c: v[key] for c, v in table.items()}).T.to_string())
    print("\n=== median box size as a fraction of the image (train) ===")
    print(pd.DataFrame({c: {"w": v["median_w_frac_of_image"], "h": v["median_h_frac_of_image"]}
                        for c, v in table.items()}).T.to_string())
    print("\nreport written to", args.out)


if __name__ == "__main__":
    main()
