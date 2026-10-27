"""A3b: apply the pHash duplicate rule (D-006 amendment) to val and test.

Stage 1:  python scripts/dup_check.py find --root <CarDD_COCO>
    Hashes val/test photos (as is and mirrored) and compares them with train; compares val with test.
    Pairs at distance <= THRESHOLD are duplicates (removed automatically in stage 2).
    Writes outputs/metrics/dup_candidates.json and a numbered spot-check sheet in figures_private.
Stage 2:  python scripts/dup_check.py apply
    Requires splits/dup_review.csv with a decision ("dup" or "keep") for every spot-check pair.
    Flags the crops of removed photos in splits/crops_manifest.csv (dup_excluded, dup_reason);
    image files on disk are never touched. Writes outputs/metrics/dup_summary.json.
"""
import argparse
import csv
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from vdv.data.dedup import hamming_matrix, phash_bits, removal_target

THRESHOLD = 6                                   # D-006 amendment: distance <= 6 -> duplicate
N_SPOT = 10                                     # closest pairs above the threshold, checked by eye
TRAIN_PAIR = ("003010.jpg", "001319.jpg")       # closest within-train pair from the calibration
HASHES = Path("/content/data/phash_train.npz")
REPO = Path(__file__).resolve().parents[1]
CANDIDATES = REPO / "outputs/metrics/dup_candidates.json"
REVIEW = REPO / "splits/dup_review.csv"
MANIFEST = REPO / "splits/crops_manifest.csv"
SUMMARY = REPO / "outputs/metrics/dup_summary.json"
SHEET = REPO / "outputs/figures_private/dup_spotcheck.png"


def names_of(root, split):
    coco = json.loads((root / "annotations" / f"instances_{split}2017.json").read_text())
    return sorted(im["file_name"] for im in coco["images"])


def load(root, split, name):
    return Image.open(root / f"{split}2017" / name).convert("RGB")


def hashes(root, split, names, mirror=False):
    return np.stack([phash_bits(ImageOps.mirror(load(root, split, n)) if mirror else load(root, split, n))
                     for n in names])


def train_hashes(root):
    names = names_of(root, "train")
    if HASHES.exists():
        z = np.load(HASHES)
        if list(z["names"]) == names and str(z["imagehash_version"]) == version("imagehash"):
            print("train hashes reused from", HASHES)
            return names, z["bits"]
        print("saved train hashes do not match the current data or imagehash version: recomputing")
    else:
        print("no saved train hashes: recomputing")
    bits = hashes(root, "train", names)
    HASHES.parent.mkdir(parents=True, exist_ok=True)
    np.savez(HASHES, names=np.array(names), bits=bits, imagehash_version=version("imagehash"))
    return names, bits


def make_pair(qs, qn, rs, rn, dist, mirrored):
    return {"query_split": qs, "query": qn, "ref_split": rs, "ref": rn,
            "distance": int(dist), "mirrored": bool(mirrored)}


def compare(q_split, q_names, q_bits, q_bits_mirror, r_split, r_names, r_bits):
    """All pairs at distance <= THRESHOLD, and each query photo's single nearest reference photo."""
    d = hamming_matrix(q_bits, r_bits).astype(int)
    mirrored = np.zeros_like(d, dtype=bool)
    if q_bits_mirror is not None:
        dm = hamming_matrix(q_bits_mirror, r_bits).astype(int)
        mirrored = dm < d
        d = np.minimum(d, dm)
    matches, nearest = [], []
    for i, qn in enumerate(q_names):
        for j in np.flatnonzero(d[i] <= THRESHOLD):
            matches.append(make_pair(q_split, qn, r_split, r_names[j], d[i, j], mirrored[i, j]))
        j = int(d[i].argmin())
        nearest.append(make_pair(q_split, qn, r_split, r_names[j], d[i, j], mirrored[i, j]))
    return matches, nearest


def spot_sheet(root, spot):
    rows, H = [], 200
    for p in spot:
        q = load(root, p["query_split"], p["query"])
        if p["mirrored"]:
            q = ImageOps.mirror(q)                      # show the orientation that matched
        r = load(root, p["ref_split"], p["ref"])
        q = q.resize((round(q.width * H / q.height), H))
        r = r.resize((round(r.width * H / r.height), H))
        row = Image.new("RGB", (q.width + r.width + 10, H + 24), "white")
        row.paste(q, (0, 24)); row.paste(r, (q.width + 10, 24))
        ImageDraw.Draw(row).text((4, 5), f"#{p['number']}  {p['query_split']}/{p['query']}"
                                 f"{' (mirrored)' if p['mirrored'] else ''}  vs  {p['ref_split']}/{p['ref']}"
                                 f"   distance {p['distance']}", fill="black")
        rows.append(row)
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height + 8 for r in rows)), "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y)); y += r.height + 8
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(SHEET)


def find(root):
    val_n, test_n = names_of(root, "val"), names_of(root, "test")
    tr_n, tr_b = train_hashes(root)
    vb, vbm = hashes(root, "val", val_n), hashes(root, "val", val_n, mirror=True)
    tb, tbm = hashes(root, "test", test_n), hashes(root, "test", test_n, mirror=True)
    print(f"hashed val {len(val_n)} and test {len(test_n)} photos (as is and mirrored)")

    comps = {"val-train": compare("val", val_n, vb, vbm, "train", tr_n, tr_b),
             "test-train": compare("test", test_n, tb, tbm, "train", tr_n, tr_b),
             "val-test": compare("val", val_n, vb, None, "test", test_n, tb)}
    auto = [m for matches, _ in comps.values() for m in matches]
    above = sorted((p for _, nearest in comps.values() for p in nearest if p["distance"] > THRESHOLD),
                   key=lambda p: (p["distance"], p["query_split"], p["query"], p["ref"]))[:N_SPOT]
    tr = dict(zip(tr_n, tr_b))
    train_pair = make_pair("train", TRAIN_PAIR[0], "train", TRAIN_PAIR[1],
                           hamming_matrix([tr[TRAIN_PAIR[0]]], [tr[TRAIN_PAIR[1]]])[0, 0], False)
    spot = [{"number": k + 1, **p} for k, p in enumerate(above + [train_pair])]

    cand = {"threshold": THRESHOLD, "n_photos": {"train": len(tr_n), "val": len(val_n), "test": len(test_n)},
            "auto_duplicates": auto, "auto_per_comparison": {c: len(m) for c, (m, _) in comps.items()},
            "spot_check": spot,
            "nearest_distance_histogram": {c: np.bincount([p["distance"] for p in n], minlength=65).tolist()
                                           for c, (_, n) in comps.items()}}
    CANDIDATES.write_text(json.dumps(cand, indent=2))
    spot_sheet(root, spot)

    print(f"\nduplicate pairs at distance <= {THRESHOLD}: {cand['auto_per_comparison']}")
    for p in auto:
        print("   auto:", p)
    print(f"\nspot-check pairs (closest above {THRESHOLD}, plus the train pair as the last number):")
    for p in spot:
        print(f"   #{p['number']:2d}  {p['query_split']}/{p['query']} vs {p['ref_split']}/{p['ref']}  "
              f"distance {p['distance']}{'  (mirrored)' if p['mirrored'] else ''}")
    print("\nwritten:", CANDIDATES, "|", SHEET)


def apply():
    cand = json.loads(CANDIDATES.read_text())
    if not REVIEW.exists():
        raise SystemExit(f"{REVIEW} missing: enter the spot-check decisions first")
    review = {int(r["number"]): r for r in csv.DictReader(open(REVIEW))}
    numbers = [p["number"] for p in cand["spot_check"]]
    undecided = [n for n in numbers if n not in review or review[n]["decision"] not in ("dup", "keep")]
    if undecided:
        raise SystemExit(f"undecided spot-check pairs: {undecided}: nothing changed")

    remove = {}
    def mark(p, how):
        target = removal_target(p)
        if target is None:
            return
        assert target[0] in ("val", "test")
        remove.setdefault(target, []).append(f"{how}: {p['ref_split'] if target == (p['query_split'], p['query']) else p['query_split']}/"
                                             f"{p['ref'] if target == (p['query_split'], p['query']) else p['query']} "
                                             f"d={p['distance']}{' mirrored' if p['mirrored'] else ''}")
    for p in cand["auto_duplicates"]:
        mark(p, "auto")
    for p in cand["spot_check"]:
        if review[p["number"]]["decision"] == "dup":
            mark(p, "spot-check")

    rows = list(csv.DictReader(open(MANIFEST)))
    fields = [f for f in rows[0].keys() if f not in ("dup_excluded", "dup_reason")] + ["dup_excluded", "dup_reason"]
    for r in rows:
        reasons = remove.get((r["split"], r["file_name"]))
        r["dup_excluded"] = int(bool(reasons))
        r["dup_reason"] = "; ".join(reasons) if reasons else ""
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    photos = Counter(s for s, _ in remove)
    crops = Counter(r["split"] for r in rows if r["dup_excluded"] == 1)
    crops_cls = Counter((r["split"], r["cls"]) for r in rows if r["dup_excluded"] == 1)
    kept = Counter(r["split"] for r in rows if r["dup_excluded"] == 0)
    decisions = [{"number": n, "decision": review[n]["decision"], "note": review[n].get("note", "")} for n in numbers]
    summary = {"threshold": cand["threshold"],
               "auto_pairs_per_comparison": cand["auto_per_comparison"],
               "photos_removed": {s: photos.get(s, 0) for s in ("val", "test")},
               "photos_removed_list": sorted(f"{s}/{n}" for s, n in remove),
               "crops_excluded": {s: crops.get(s, 0) for s in ("train", "val", "test")},
               "crops_excluded_per_class": {f"{s}/{c}": n for (s, c), n in sorted(crops_cls.items())},
               "crops_remaining": dict(kept),
               "spot_check": {"pairs": len(numbers), "dup": sum(d["decision"] == "dup" for d in decisions),
                              "keep": sum(d["decision"] == "keep" for d in decisions), "decisions": decisions}}
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "spot_check"}, indent=2))
    print("spot-check:", {k: v for k, v in summary["spot_check"].items() if k != "decisions"})
    print("written:", MANIFEST, "|", SUMMARY)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["find", "apply"])
    ap.add_argument("--root", type=Path, help="Path to CarDD_COCO (needed for find)")
    a = ap.parse_args()
    find(a.root) if a.stage == "find" else apply()
