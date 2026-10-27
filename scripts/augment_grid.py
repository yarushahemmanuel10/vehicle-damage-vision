"""A5: private example grid of train-time augmentation (D-021). One train crop per class:
the original, then 4 augmented draws, each padded to 224 x 224.
Writes outputs/figures_private/augment_grid_train.png (git-ignored: dataset images)."""
from pathlib import Path

import cv2
import numpy as np

from vdv.data.augment import augment_train
from vdv.data.crops import check_crop_cache, crop_code_version, pad_to_square
from vdv.data.manifest import CLASSES, load_split

REPO = Path(__file__).resolve().parents[1]
CROP_DIR = Path("/content/data") / f"crops_{crop_code_version()}"
OUT = REPO / "outputs/figures_private/augment_grid_train.png"


def tile(img, text):
    strip = np.full((22, 224, 3), 255, np.uint8)
    cv2.putText(strip, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([pad_to_square(img), strip])


def main():
    check_crop_cache(CROP_DIR)
    rows = load_split("train")
    rng = np.random.default_rng(42)
    grid = []
    for cls in CLASSES:
        r = min((r for r in rows if r["cls"] == cls), key=lambda r: int(r["ann_id"]))
        img = cv2.imread(str(CROP_DIR / r["crop_path"]))
        tiles = [tile(img, f"{cls} #{r['ann_id']} original")]
        for _ in range(4):
            aug, p = augment_train(img, rng)
            tiles.append(tile(aug, f"{'flip' if p['flipped'] else 'no flip'}, x{p['brightness']:.2f}"))
        grid.append(np.hstack(tiles))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), np.vstack(grid))
    print("written:", OUT)


if __name__ == "__main__":
    main()
