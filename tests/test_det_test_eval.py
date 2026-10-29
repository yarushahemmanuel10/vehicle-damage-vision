import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("eval_det_test", REPO / "scripts" / "eval_det_test.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

CK = {"epoch": 12, "seed": 42, "git_commit": "abc"}


def _metrics(**kw):
    m = {"finished": True, "best_epoch": 12, "seed": 42, "git_commit": "abc", "checkpoint": "/ck/best.pt",
         "history": [{"epoch": 11, "val": {"mAP_50_95": 0.49}}, {"epoch": 12, "val": {"mAP_50_95": 0.4931}}]}
    m.update(kw)
    return m


def test_refuse_if_done(tmp_path):
    paths = [tmp_path / "local.json", tmp_path / "drive.json"]
    E.refuse_if_done(paths)
    paths[1].write_text("{}")
    with pytest.raises(RuntimeError, match="used once"):
        E.refuse_if_done(paths)


def test_refuse_if_dirty():
    E.refuse_if_dirty("")
    for porcelain in [" M src/vdv/det_eval.py\n", "?? scripts/new.py\n"]:
        with pytest.raises(RuntimeError, match="uncommitted"):
            E.refuse_if_dirty(porcelain)


def test_check_checkpoint_returns_best_row():
    assert E.check_checkpoint(CK, _metrics(), "/ck/best.pt")["epoch"] == 12


@pytest.mark.parametrize("ck,metrics,path", [
    ({**CK, "epoch": 11}, _metrics(), "/ck/best.pt"),
    ({**CK, "git_commit": "zzz"}, _metrics(), "/ck/best.pt"),
    (CK, _metrics(finished=False), "/ck/best.pt"),
    (CK, _metrics(), "/other/best.pt"),
])
def test_check_checkpoint_mismatch(ck, metrics, path):
    with pytest.raises(RuntimeError, match="does not match"):
        E.check_checkpoint(ck, metrics, path)


def test_check_val_tolerance():
    assert E.check_val(0.4925, 0.4931) == pytest.approx(0.0006)
    E.check_val(0.4949, 0.4931)                       # 0.0018, inside
    for bad in (0.4955, 0.4900):                      # 0.0024 and 0.0031, outside
        with pytest.raises(RuntimeError, match="NOT touched"):
            E.check_val(bad, 0.4931)


def test_comparison_rows():
    test = {k: 0.5 for k in E.METRICS}
    test["per_class_AP"] = {c: 0.5 for c in E.PAPER["per_class_AP"]}
    rows = {r["metric"]: r for r in E.comparison(test)}
    assert len(rows) == len(E.METRICS) + 6
    assert rows["mAP_50_95"]["diff"] == pytest.approx(0.5 - 0.492)
    assert rows["AP crack"]["diff"] == pytest.approx(0.5 - 0.195)
    assert rows["mAP_75"]["paper"] is None and rows["mAP_75"]["diff"] is None


def test_paper_classes_match_run():
    m = json.loads(E.RUN_METRICS.read_text())
    best = [h for h in m["history"] if h["epoch"] == m["best_epoch"]][0]
    assert set(E.PAPER["per_class_AP"]) == set(best["val"]["per_class_AP"])
