"""A2: build one classification crop per annotation for all splits (D-014..D-016).

Usage:
    python scripts/build_crops.py --root /content/data/CarDD_release/CarDD_COCO \
        --out /content/data --drive /content/drive/MyDrive/vdv/data
Writes:
    <out>/crops_<version>/<split>/<class>/<ann_id>.png   crops (aspect kept, long side 224, unpadded)
    <out>/crops_<version>/meta.json                        version, parameters, counts
    <drive>/crops_<version>.zip                            persistent copy (if --drive is given)
    splits/crops_manifest.csv                              one row per crop, keyed on (split, ann_id)
    outputs/metrics/crops_summary.json                     counts per split and class
The manifest is provisional until A4 (after the A3 duplicate check).
"""
import argparse
import csv
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from vdv.data.boxes import intersection_area
from vdv.data.crops import (MARGIN, MAX_ASPECT, MIN_SIDE, STORE_LONG_SIDE, check_crop_cache,
                            crop_code_version, crop_window, resize_long_side)

SPLITS = ["train", "val", "test"]
REPO = Path(__file__).resolve().parents[1]
FLAGS = ["min_window_applied", "aspect_capped", "aspect_cap_incomplete",
         "other_class_any", "other_class_ge25", "other_class_ge50"]
FIELDS = ["split", "ann_id", "image_id", "file_name", "cls", "category_id",
          "box_x", "box_y", "box_w", "box_h", "win_x0", "win_y0", "win_x1", "win_y1",
          "win_w", "win_h", "win_aspect", *FLAGS, "other_class_max_share",
          "crop_path", "crop_w", "crop_h"]


def slug(name):
    return name.replace(" ", "_")


def id_uniqueness(cocos):
    ann = Counter(a["id"] for s in SPLITS for a in cocos[s]["annotations"])
    img = Counter(im["id"] for s in SPLITS for im in cocos[s]["images"])
    return {"annotation_ids_unique_across_splits": all(c == 1 for c in ann.values()),
            "annotation_ids_shared_between_splits": sum(1 for c in ann.values() if c > 1),
            "image_ids_unique_across_splits": all(c == 1 for c in img.values()),
            "image_ids_shared_between_splits": sum(1 for c in img.values() if c > 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="Path to CarDD_COCO")
    ap.add_argument("--out", type=Path, default=Path("/content/data"))
    ap.add_argument("--drive", type=Path, default=None, help="Folder on Drive for the zip")
    args = ap.parse_args()

    version = crop_code_version()
    cocos = {s: json.loads((args.root / "annotations" / f"instances_{s}2017.json").read_text())
             for s in SPLITS}
    ids = id_uniqueness(cocos)

    crop_root = args.out / f"crops_{version}"
    if crop_root.exists():
        shutil.rmtree(crop_root)                       # rebuild cleanly (local folder only)
    crop_root.mkdir(parents=True)

    rows = []
    for s in SPLITS:
        coco = cocos[s]
        cat = {c["id"]: c["name"] for c in coco["categories"]}
        by_image = defaultdict(list)
        for a in coco["annotations"]:
            by_image[a["image_id"]].append(a)
        for im in coco["images"]:
            img = cv2.imread(str(args.root / f"{s}2017" / im["file_name"]))
            assert img is not None, f"cannot read {im['file_name']}"
            for a in by_image[im["id"]]:
                (x0, y0, x1, y1), flags = crop_window(a["bbox"], im["width"], im["height"])
                bx, by, bw, bh = a["bbox"]
                assert 0 <= x0 <= bx and 0 <= y0 <= by and bx + bw <= x1 <= im["width"] \
                    and by + bh <= y1 <= im["height"], f"bad window for ann {a['id']}"

                # D-014: largest share of any OTHER-class box that lies inside the FINAL window
                window = (x0, y0, x1 - x0, y1 - y0)
                share = 0.0
                for b in by_image[im["id"]]:
                    if b["id"] != a["id"] and b["category_id"] != a["category_id"]:
                        share = max(share, intersection_area(window, b["bbox"]) / (b["bbox"][2] * b["bbox"][3]))

                crop = resize_long_side(img[y0:y1, x0:x1])
                rel = Path(s) / slug(cat[a["category_id"]]) / f"{a['id']}.png"
                (crop_root / rel).parent.mkdir(parents=True, exist_ok=True)
                assert cv2.imwrite(str(crop_root / rel), crop), f"cannot write {rel}"

                rows.append({"split": s, "ann_id": a["id"], "image_id": im["id"], "file_name": im["file_name"],
                             "cls": cat[a["category_id"]], "category_id": a["category_id"],
                             "box_x": round(bx, 2), "box_y": round(by, 2), "box_w": round(bw, 2), "box_h": round(bh, 2),
                             "win_x0": x0, "win_y0": y0, "win_x1": x1, "win_y1": y1,
                             "win_w": x1 - x0, "win_h": y1 - y0,
                             "win_aspect": round(max(x1 - x0, y1 - y0) / min(x1 - x0, y1 - y0), 3),
                             **{k: int(v) for k, v in flags.items()},
                             "other_class_any": int(share > 0), "other_class_ge25": int(share >= 0.25),
                             "other_class_ge50": int(share >= 0.50), "other_class_max_share": round(share, 4),
                             "crop_path": rel.as_posix(), "crop_w": crop.shape[1], "crop_h": crop.shape[0]})

    # --- Integrity checks ---
    keys = [(r["split"], r["ann_id"]) for r in rows]
    assert len(keys) == len(set(keys)), "(split, ann_id) is not unique"
    n_ann = sum(len(cocos[s]["annotations"]) for s in SPLITS)
    assert len(rows) == n_ann, f"{len(rows)} crops for {n_ann} annotations"
    verification = REPO / "outputs/metrics/data_verification.json"
    if verification.exists():
        assert len(rows) == json.loads(verification.read_text())["total_instances"]
    rows.sort(key=lambda r: (SPLITS.index(r["split"]), r["ann_id"]))

    # --- Summary per split and class ---
    summary = {"version": version, "n_crops": len(rows), "ids": ids,
               "parameters": {"margin": MARGIN, "min_side_px": MIN_SIDE, "max_aspect": MAX_ASPECT,
                              "store_long_side_px": STORE_LONG_SIDE},
               "per_split": {}}
    for s in SPLITS:
        per_cls = {}
        for c in sorted({r["cls"] for r in rows if r["split"] == s}):
            sub = [r for r in rows if r["split"] == s and r["cls"] == c]
            per_cls[c] = {"crops": len(sub), **{f: sum(r[f] for r in sub) for f in FLAGS}}
        tot = [r for r in rows if r["split"] == s]
        per_cls["ALL"] = {"crops": len(tot), **{f: sum(r[f] for r in tot) for f in FLAGS}}
        summary["per_split"][s] = per_cls

    # --- Write manifest, metadata, summary ---
    manifest = REPO / "splits" / "crops_manifest.csv"
    for path in [manifest, crop_root / "crops_manifest.csv"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    (crop_root / "meta.json").write_text(json.dumps(summary, indent=2))
    (REPO / "outputs/metrics/crops_summary.json").write_text(json.dumps(summary, indent=2))
    check_crop_cache(crop_root)                        # the cache we just built must be accepted

    # --- Zip (stored, no compression: PNGs are already compressed) and copy to Drive ---
    if args.drive:
        local_zip = args.out / f"crops_{version}.zip"
        with zipfile.ZipFile(local_zip, "w", compression=zipfile.ZIP_STORED) as z:
            for p in sorted(crop_root.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(args.out))
        args.drive.mkdir(parents=True, exist_ok=True)
        shutil.copy(local_zip, args.drive / local_zip.name)
        local_zip.unlink()                             # deletes the LOCAL zip only
        print(f"zip copied to {args.drive / local_zip.name} "
              f"({(args.drive / local_zip.name).stat().st_size / 1e6:.0f} MB)")

    # --- Human-readable summary ---
    print("version:", version, "| crops:", len(rows), "| ids:", ids)
    for s in SPLITS:
        print(f"\n[{s}]")
        print(f"  {'class':14s} {'crops':>6s} " + " ".join(f"{f:>22s}" for f in FLAGS))
        for c, v in summary["per_split"][s].items():
            print(f"  {c:14s} {v['crops']:6d} " + " ".join(f"{v[f]:22d}" for f in FLAGS))
    print("\nmanifest:", manifest, "| summary:", REPO / "outputs/metrics/crops_summary.json")


if __name__ == "__main__":
    main()
