"""A4: the classification split is frozen. Reads the committed manifest only (no images needed)."""
import csv
import json
from collections import defaultdict

import pytest

from vdv.data.manifest import CLASSES, FROZEN_HASH, MANIFEST, REPO_ROOT, load_split, manifest_sha256


def rows():
    with open(MANIFEST, newline="") as f:
        return list(csv.DictReader(f))


def test_manifest_matches_frozen_hash():
    assert manifest_sha256() == FROZEN_HASH.read_text().split()[0], "manifest changed after the A4 freeze"


def test_key_split_ann_id_is_unique():
    keys = [(r["split"], r["ann_id"]) for r in rows()]
    assert len(keys) == len(set(keys))


def test_no_photo_in_two_splits():
    splits_of = defaultdict(set)
    for r in rows():
        splits_of[r["file_name"]].add(r["split"])
    assert all(len(s) == 1 for s in splits_of.values())


def test_only_known_splits():
    assert {r["split"] for r in rows()} == {"train", "val", "test"}


def test_train_never_excluded_and_every_exclusion_has_a_reason():
    for r in rows():
        excluded = r["dup_excluded"] == "1"
        assert not (excluded and r["split"] == "train")
        assert excluded == bool(r["dup_reason"])


def test_excluded_photo_loses_all_its_crops():
    flags = defaultdict(set)
    for r in rows():
        flags[(r["split"], r["file_name"])].add(r["dup_excluded"])
    assert all(len(v) == 1 for v in flags.values())


def test_remaining_counts_match_dup_summary():
    summary = json.loads((REPO_ROOT / "outputs/metrics/dup_summary.json").read_text())
    kept = defaultdict(int)
    for r in rows():
        kept[r["split"]] += r["dup_excluded"] == "0"
    assert dict(kept) == summary["crops_remaining"]


def test_label_order_matches_category_ids():
    for r in rows():
        assert CLASSES[int(r["category_id"]) - 1] == r["cls"]


def test_load_split_excludes_duplicates():
    summary = json.loads((REPO_ROOT / "outputs/metrics/dup_summary.json").read_text())
    for split in ["train", "val", "test"]:
        got = load_split(split)
        assert len(got) == summary["crops_remaining"][split]
        assert all(r["dup_excluded"] == "0" for r in got)


def test_load_split_refuses_modified_manifest(tmp_path):
    changed = tmp_path / "crops_manifest.csv"
    changed.write_bytes(MANIFEST.read_bytes() + b"\n")
    with pytest.raises(RuntimeError):
        load_split("train", manifest=changed)
