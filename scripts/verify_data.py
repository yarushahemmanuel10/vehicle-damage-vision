"""A1: verify CarDD's COCO data on disk against its annotation files (read-only).

Usage:
    python scripts/verify_data.py --root /content/data/CarDD_release/CarDD_COCO
Writes outputs/metrics/data_verification.json (file names and counts only, no images).
"""
import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image

from vdv.data.checks import box_overshoot, size_check
from vdv.data.image_info import load_image_info

SPLITS = ["train", "val", "test"]
EXIF_ORIENTATION_TAG = 0x0112        # 1 = normal; other values ask viewers to rotate/flip
TINY_OVERSHOOT_PX = 1.0              # boxes outside the image by <= 1 px count as "tiny"


def counts(series):
    """value_counts as a plain {str: int} dict (JSON-safe); blank cells shown as 'blank'."""
    return {("blank" if pd.isna(k) else str(k)): int(v)
            for k, v in series.value_counts(dropna=False).items()}


def verify_split(root, split):
    coco = json.loads((root / "annotations" / f"instances_{split}2017.json").read_text())
    img_dir = root / f"{split}2017"
    images = {im["id"]: im for im in coco["images"]}
    names_json = [im["file_name"] for im in coco["images"]]
    names_disk = {p.name for p in img_dir.glob("*.jpg")}
    categories = sorted((c["id"], c["name"]) for c in coco["categories"])
    cat_name = dict(categories)

    r = {
        "n_images_json": len(names_json),
        "n_images_disk": len(names_disk),
        "n_annotations": len(coco["annotations"]),
        "categories": categories,
        "duplicate_names_in_json": sorted(n for n, c in Counter(names_json).items() if c > 1),
        "duplicate_image_ids": sorted(i for i, c in Counter(im["id"] for im in coco["images"]).items() if c > 1),
        "missing_on_disk": sorted(set(names_json) - names_disk),
        "on_disk_not_in_json": sorted(names_disk - set(names_json)),
    }

    # --- Every image: loads with OpenCV? size matches JSON? EXIF orientation tag? ---
    load_failed, exif_tagged = [], []
    size_results = defaultdict(list)
    for im in coco["images"]:
        path = img_dir / im["file_name"]
        if not path.exists():
            continue
        arr = cv2.imread(str(path))              # OpenCV's default applies EXIF orientation
        if arr is None:
            load_failed.append(im["file_name"])
            continue
        json_wh = (im["width"], im["height"])
        opencv_wh = (arr.shape[1], arr.shape[0])
        with Image.open(path) as pil:
            raw_wh = pil.size                    # stored pixels, no rotation applied
            orientation = pil.getexif().get(EXIF_ORIENTATION_TAG, 1)
        if orientation != 1:
            exif_tagged.append(im["file_name"])
        size_results["json_vs_opencv_" + size_check(json_wh, opencv_wh)].append(im["file_name"])
        size_results["json_vs_raw_" + size_check(json_wh, raw_wh)].append(im["file_name"])

    r["load_failed"] = load_failed
    r["exif_orientation_tagged"] = exif_tagged
    r["size_counts"] = {k: len(v) for k, v in sorted(size_results.items())}
    r["size_mismatch_files"] = {k: sorted(v) for k, v in size_results.items() if not k.endswith("match")}
    swaps = set(size_results.get("json_vs_opencv_swap", []))
    r["opencv_swaps_with_exif_tag"] = len(swaps & set(exif_tagged))

    # --- Every annotation: valid category and image, non-zero area, inside the image ---
    valid_ids = set(cat_name)
    per_category = Counter()
    zero_area, tiny_over, real_over, bad_category, bad_image = [], [], [], [], []
    max_overshoot = 0.0
    seg_types = Counter()
    for a in coco["annotations"]:
        seg = a.get("segmentation")
        if isinstance(seg, list) and any(len(poly) >= 6 for poly in seg):
            seg_types["polygon"] += 1          # at least one polygon with 3+ points
        elif isinstance(seg, dict) and seg.get("counts"):
            seg_types["rle"] += 1              # run-length encoded mask
        if a["category_id"] not in valid_ids:
            bad_category.append(a["id"])
            continue
        im = images.get(a["image_id"])
        if im is None:
            bad_image.append(a["id"])
            continue
        per_category[cat_name[a["category_id"]]] += 1
        _, _, w, h = a["bbox"]
        if w <= 0 or h <= 0:
            zero_area.append(a["id"])
        over = box_overshoot(a["bbox"], im["width"], im["height"])
        max_overshoot = max(max_overshoot, over)
        if 0 < over <= TINY_OVERSHOOT_PX:
            tiny_over.append(a["id"])
        elif over > TINY_OVERSHOOT_PX:
            real_over.append(a["id"])

    annotated = {a["image_id"] for a in coco["annotations"]}
    r.update({
        "instances_per_category": dict(per_category.most_common()),
        "annotations_bad_category": bad_category,
        "annotations_bad_image_id": bad_image,
        "boxes_zero_area": zero_area,
        "boxes_tiny_overshoot": tiny_over,
        "boxes_real_overshoot": real_over,
        "max_box_overshoot_px": round(max_overshoot, 3),
        "annotations_with_segmentation": sum(seg_types.values()),
        "segmentation_types": dict(seg_types),
        "images_without_annotations": sorted(images[i]["file_name"] for i in set(images) - annotated),
    })
    return r, coco


def check_image_info(root, coco_by_split):
    """D-012: spreadsheet summary, bins per split, and cross-check against the annotations."""
    df = load_image_info(root / "annotations" / "image_info.xlsx")
    split_of = {im["file_name"]: s for s, coco in coco_by_split.items() for im in coco["images"]}
    df["split"] = df["file_name"].map(split_of)

    info = {
        "n_rows": int(len(df)),
        "columns_after_strip": list(df.columns),
        "rows_not_in_any_split": int(df["split"].isna().sum()),
        "duplicate_file_names": int(df["file_name"].duplicated().sum()),
        "raw_value_counts": {c: counts(df[c]) for c in ["shooting angle", "color", "complete or partial"]},
        "bins_all": {b: counts(df[b]) for b in ["colour_bin", "angle_bin", "completeness"]},
        "bins_per_split": {s: {b: counts(df.loc[df["split"] == s, b])
                               for b in ["colour_bin", "angle_bin", "completeness"]} for s in SPLITS},
    }
    info["partial_share"] = round(float((df["completeness"] == "partial").mean()), 4)

    # Cross-check width / height / #instances / #categories against the COCO annotations
    rows = df.set_index("file_name")
    mismatches = {"width": [], "height": [], "#instances": [], "#categories": []}
    for coco in coco_by_split.values():
        n_inst = Counter(a["image_id"] for a in coco["annotations"])
        cats = defaultdict(set)
        for a in coco["annotations"]:
            cats[a["image_id"]].add(a["category_id"])
        for im in coco["images"]:
            row = rows.loc[im["file_name"]]
            for field, value in [("width", im["width"]), ("height", im["height"]),
                                 ("#instances", n_inst[im["id"]]), ("#categories", len(cats[im["id"]]))]:
                if int(row[field]) != value:
                    mismatches[field].append(im["file_name"])
    info["cross_check_mismatches"] = {k: sorted(v) for k, v in mismatches.items()}
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="Path to CarDD_COCO")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "outputs/metrics/data_verification.json")
    args = ap.parse_args()

    report, coco_by_split = {"splits": {}}, {}
    for split in SPLITS:
        report["splits"][split], coco_by_split[split] = verify_split(args.root, split)
    names = {s: {im["file_name"] for im in coco_by_split[s]["images"]} for s in SPLITS}

    report["categories_identical_across_splits"] = len(
        {tuple(map(tuple, report["splits"][s]["categories"])) for s in SPLITS}) == 1
    report["file_names_in_two_splits"] = {
        f"{a}-{b}": sorted(names[a] & names[b]) for a, b in combinations(SPLITS, 2)}
    report["total_images"] = sum(report["splits"][s]["n_images_json"] for s in SPLITS)
    report["total_instances"] = sum(report["splits"][s]["n_annotations"] for s in SPLITS)
    total_per_cat = Counter()
    for s in SPLITS:
        total_per_cat.update(report["splits"][s]["instances_per_category"])
    report["instances_per_category_all"] = dict(total_per_cat.most_common())
    report["image_info"] = check_image_info(args.root, coco_by_split)
    report["settings"] = {"tiny_overshoot_px": TINY_OVERSHOOT_PX}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    # --- Human-readable summary ---
    for s in SPLITS:
        r = report["splits"][s]
        print(f"[{s}] images json/disk {r['n_images_json']}/{r['n_images_disk']} | "
              f"annotations {r['n_annotations']} | missing {len(r['missing_on_disk'])} | "
              f"extra {len(r['on_disk_not_in_json'])} | dup names {len(r['duplicate_names_in_json'])} | "
              f"dup ids {len(r['duplicate_image_ids'])} | load failed {len(r['load_failed'])}")
        print(f"      sizes {r['size_counts']} | EXIF-tagged {len(r['exif_orientation_tagged'])} "
              f"(of which OpenCV swaps: {r['opencv_swaps_with_exif_tag']})")
        print(f"      boxes: zero-area {len(r['boxes_zero_area'])}, tiny overshoot {len(r['boxes_tiny_overshoot'])}, "
              f"real overshoot {len(r['boxes_real_overshoot'])}, max overshoot {r['max_box_overshoot_px']} px | "
              f"bad category {len(r['annotations_bad_category'])}, bad image id {len(r['annotations_bad_image_id'])}, "
              f"images without annotations {len(r['images_without_annotations'])}")
        print(f"      instances per category: {r['instances_per_category']}")
        print(f"      segmentation: {r['annotations_with_segmentation']}/{r['n_annotations']} "
              f"annotations have a non-empty mask {r['segmentation_types']}")
    print("categories identical across splits:", report["categories_identical_across_splits"])
    print("file names in two splits:", {k: len(v) for k, v in report["file_names_in_two_splits"].items()})
    print(f"TOTAL: {report['total_images']} images, {report['total_instances']} instances:",
          report["instances_per_category_all"])

    ii = report["image_info"]
    print(f"\n[image_info.xlsx] rows {ii['n_rows']} | not in any split {ii['rows_not_in_any_split']} | "
          f"duplicate file names {ii['duplicate_file_names']} | partial share {ii['partial_share']}")
    print("      cross-check mismatches:", {k: len(v) for k, v in ii["cross_check_mismatches"].items()})
    for b, c in ii["bins_all"].items():
        print(f"      {b}: {c}")
    for s in SPLITS:
        print(f"      [{s}] " + " | ".join(f"{b}: {c}" for b, c in ii["bins_per_split"][s].items()))
    print("report written to", args.out)


if __name__ == "__main__":
    main()
