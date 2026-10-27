import numpy as np
import pytest
from sklearn.metrics import f1_score

from vdv.evaluation import assert_not_evaluated, bootstrap_by_photo, macro_f1, per_class


def test_macro_f1_matches_sklearn_when_all_classes_present():
    rng = np.random.default_rng(0)
    y = np.concatenate([np.arange(6), rng.integers(0, 6, 300)])
    pred = np.where(rng.random(len(y)) < 0.7, y, rng.integers(0, 6, len(y)))
    assert abs(macro_f1(y, pred, 6) - f1_score(y, pred, average="macro", labels=list(range(6)))) < 1e-12


def test_per_class_counts_support():
    _, _, _, support, cm = per_class([0, 0, 1, 2], [0, 1, 1, 2], 3)
    assert support.tolist() == [2, 1, 1] and cm.sum() == 4


def test_bootstrap_resamples_photos_not_crops():
    photos = ["A", "A", "A", "B"]                       # photo A: 3 correct crops; photo B: 1 wrong crop
    y, pred = [0, 0, 0, 1], [0, 0, 0, 0]
    out = bootstrap_by_photo(photos, y, {"m": pred}, 2, n_resamples=300, return_samples=True)
    # draws AA, AB or BB give accuracy 1.0, 0.75 or 0.0; crop-level resampling would also give e.g. 0.5
    assert set(out["samples"]["m"]["accuracy"]) <= {1.0, 0.75, 0.0}


def test_paired_difference_of_identical_models_is_zero():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 60)
    pred = rng.integers(0, 3, 60)
    out = bootstrap_by_photo([f"p{i // 2}" for i in range(60)], y, {"a": pred, "b": pred.copy()}, 3,
                             n_resamples=200, reference="a")
    assert out["paired_differences"]["a minus b"]["macro_f1"] == [0.0, 0.0]


def test_run_once_guard(tmp_path):
    assert_not_evaluated(tmp_path / "test_results.json")          # nothing yet: allowed
    (tmp_path / "test_results.json").write_text("{}")
    with pytest.raises(RuntimeError):
        assert_not_evaluated(tmp_path / "test_results.json")
