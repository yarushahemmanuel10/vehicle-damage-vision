"""A3 step 1: calibrate the pHash duplicate threshold on TRAIN only (D-006, D-013).

For a fixed random sample of train photos, make synthetic near-duplicates and measure their pHash
distance to the original; for the same photos, measure the distance to the nearest OTHER train photo.

Usage:
    python scripts/dup_calibrate.py --root /content/data/CarDD_release/CarDD_COCO
Writes:
    outputs/metrics/dup_calibration_train.json
    outputs/figures/dup_calibration_train.png    (a histogram only: no dataset images)
    /content/data/phash_train.npz                (all train hashes, keyed by file name; reused by A3b)
"""
import argparse
import io
import json
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance

from vdv.data.dedup import hamming_matrix, phash_bits
from vdv.seed import set_seed

N_SAMPLE, SEED = 200, 42
HASHES = Path("/content/data/phash_train.npz")          # outside the repo
REPO = Path(__file__).resolve().parents[1]


def jpeg(img, quality=70):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def resize_half(img):
    w, h = img.size
    return img.resize((w // 2, h // 2), Image.BILINEAR)


def crop(img, rng):
    """Remove 5-10% of each dimension, split randomly between the two sides."""
    w, h = img.size
    frac = rng.uniform(0.05, 0.10)
    dx, dy = int(w * frac), int(h * frac)
    ox, oy = int(rng.integers(0, dx + 1)), int(rng.integers(0, dy + 1))
    return img.crop((ox, oy, w - (dx - ox), h - (dy - oy)))


def brightness(img, rng):
    return ImageEnhance.Brightness(img).enhance(rng.uniform(0.9, 1.1))


def summary(values):
    v = np.asarray(values)
    return {"n": int(len(v)), "min": int(v.min()), "p50": float(np.percentile(v, 50)),
            "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99)), "max": int(v.max())}


def histogram(values):
    return np.bincount(np.asarray(values), minlength=65).tolist()       # counts for distances 0..64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="Path to CarDD_COCO")
    args = ap.parse_args()
    set_seed(SEED)
    rng = np.random.default_rng(SEED)

    coco = json.loads((args.root / "annotations" / "instances_train2017.json").read_text())
    names = sorted(im["file_name"] for im in coco["images"])
    load = lambda n: Image.open(args.root / "train2017" / n).convert("RGB")
    all_bits = np.stack([phash_bits(load(n)) for n in names])
    print(f"hashed {len(names)} train photos")
    HASHES.parent.mkdir(parents=True, exist_ok=True)
    np.savez(HASHES, names=np.array(names), bits=all_bits, imagehash_version=version("imagehash"))
    print("train hashes saved to", HASHES)

    sample = sorted(int(i) for i in rng.choice(len(names), size=N_SAMPLE, replace=False))
    synthetic = {"jpeg_q70": [], "resize_half": [], "crop_5_10pct": [], "brightness_10pct": [], "combined": []}
    for i in sample:
        img, orig = load(names[i]), all_bits[i]
        copies = {"jpeg_q70": jpeg(img), "resize_half": resize_half(img),
                  "crop_5_10pct": crop(img, rng), "brightness_10pct": brightness(img, rng),
                  "combined": brightness(crop(resize_half(jpeg(img)), rng), rng)}
        for k, c in copies.items():
            synthetic[k].append(int(hamming_matrix([phash_bits(c)], [orig])[0, 0]))

    d = hamming_matrix(all_bits[sample], all_bits).astype(int)
    d[np.arange(N_SAMPLE), sample] = 999                                 # exclude each photo itself
    nearest_idx = d.argmin(axis=1)
    nearest = [int(x) for x in d[np.arange(N_SAMPLE), nearest_idx]]

    all_synth = [x for v in synthetic.values() for x in v]
    max_synth = max(all_synth)
    close_pairs = sorted(({"sample": names[s], "nearest": names[int(j)], "distance": dist}
                          for s, j, dist in zip(sample, nearest_idx, nearest) if dist <= max_synth),
                         key=lambda p: p["distance"])

    result = {"split": "train", "n_train_photos": len(names), "n_sample": N_SAMPLE, "seed": SEED,
              "hash": f"imagehash.phash, 64 bits (imagehash {version('imagehash')})",
              "synthetic": {k: summary(v) for k, v in synthetic.items()},
              "synthetic_all": summary(all_synth), "nearest_other": summary(nearest),
              "histogram_synthetic_all": histogram(all_synth), "histogram_nearest_other": histogram(nearest),
              "nearest_other_within_synthetic_range": close_pairs}
    out = REPO / "outputs/metrics/dup_calibration_train.json"
    out.write_text(json.dumps(result, indent=2))

    fig, ax = plt.subplots(figsize=(9, 4))
    bins = np.arange(66) - 0.5
    ax.hist(all_synth, bins=bins, alpha=0.6, label=f"synthetic near-duplicate vs original (n={len(all_synth)})")
    ax.hist(nearest, bins=bins, alpha=0.6, label=f"nearest OTHER train photo (n={len(nearest)})")
    ax.set_xlabel("pHash Hamming distance (0 = identical fingerprints, 64 = all bits differ)")
    ax.set_ylabel("count")
    ax.set_title("Duplicate-threshold calibration, train only (D-006)")
    ax.legend()
    fig.tight_layout()
    fig_path = REPO / "outputs/figures/dup_calibration_train.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=120)

    print(f"\n{'':18s} {'n':>5s} {'min':>4s} {'p50':>6s} {'p95':>6s} {'p99':>6s} {'max':>4s}")
    rows = {**result["synthetic"], "SYNTHETIC (all)": result["synthetic_all"],
            "NEAREST OTHER": result["nearest_other"]}
    for k, s in rows.items():
        print(f"{k:18s} {s['n']:5d} {s['min']:4d} {s['p50']:6.1f} {s['p95']:6.1f} {s['p99']:6.1f} {s['max']:4d}")

    hs, hn = result["histogram_synthetic_all"], result["histogram_nearest_other"]
    print("\ndistance | synthetic | nearest-other")
    for dist in range(0, min(65, max(max_synth, min(nearest)) + 6)):
        print(f"{dist:8d} | {hs[dist]:9d} | {hn[dist]:13d}")

    print(f"\nnearest-other pairs within the synthetic range (distance <= {max_synth}): {len(close_pairs)}")
    for p in close_pairs[:20]:
        print("  ", p)
    print("written:", out, "|", fig_path)


if __name__ == "__main__":
    main()
