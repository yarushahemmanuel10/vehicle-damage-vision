"""The frozen classification split (A4): splits/crops_manifest.csv, verified against its SHA-256."""
import csv
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = REPO_ROOT / "splits" / "crops_manifest.csv"
FROZEN_HASH = REPO_ROOT / "splits" / "crops_manifest.sha256"

# Label index = CarDD category_id - 1 (fixed order, never sorted or inferred from data)
CLASSES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]


def manifest_sha256(path=MANIFEST):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_split(split, manifest=MANIFEST, frozen_hash=FROZEN_HASH):
    """Rows of one split with duplicates excluded (dup_excluded == 0).
    Refuses a manifest that differs from the frozen hash."""
    expected = Path(frozen_hash).read_text().split()[0]
    if manifest_sha256(manifest) != expected:
        raise RuntimeError(f"{manifest} differs from the frozen hash in {frozen_hash}: "
                           "the split was changed after the A4 freeze")
    with open(manifest, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split and r["dup_excluded"] == "0"]
    if not rows:
        raise ValueError(f"no rows for split {split!r}")
    return rows
