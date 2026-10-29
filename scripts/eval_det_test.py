"""B5: one-off detection test evaluation of the B4 checkpoint (D-026, D-028, D-029).

Runs once. Refuses if the results already exist (locally or on Drive), if the repo has
uncommitted changes, or if the checkpoint does not match the run's metrics.json.
Before touching the test set it re-evaluates validation and stops if mAP@50:95
differs from the recorded value by more than VAL_TOL.
"""
import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN = "fasterrcnn_r50_fpn_v1_seed42"
RUN_METRICS = REPO / "outputs" / "metrics" / "runs" / RUN / "metrics.json"
RESULTS = REPO / "outputs" / "metrics" / "det_test_results.json"
DRIVE = Path("/content/drive/MyDrive/vdv")
DRIVE_RESULTS = DRIVE / "metrics" / "det_test_results.json"
DETS = REPO / "outputs" / "private" / "det_test_detections.json"      # git-ignored (D-027)
DRIVE_DETS = DRIVE / "private" / "det_test_detections.json"
VAL_TOL = 0.002          # AMP inference is not bit-exact (D-029)
N_TEST = 374
SEED = 42

# Verified in D-026: Wang et al., CarDD, arXiv 2211.00945, Tables IV-V (Mask R-CNN R50, box AP, test set)
PAPER = {
    "source": "Wang et al., CarDD, arXiv 2211.00945, Tables IV-V: Mask R-CNN R50, box AP, CarDD test set",
    "mAP_50_95": 0.492, "mAP_50": 0.663, "mAP_75": None,
    "AP_small": 0.162, "AP_medium": 0.418, "AP_large": 0.559,
    "per_class_AP": {"dent": 0.270, "scratch": 0.253, "crack": 0.195,
                     "glass shatter": 0.870, "lamp broken": 0.538, "tire flat": 0.826},
}
CAVEATS = [
    "Same 374 test images as the paper (official CarDD test split).",
    "The checkpoint was chosen on 805 validation photos; the official split has 810 (5 removed as duplicates, D-006).",
    "Trained for 12 epochs (D-028); the paper trained for 24.",
    "The paper's Mask R-CNN trains boxes jointly with masks; this model is a box-only Faster R-CNN.",
    "AP75 was not recorded from the paper (D-026), so it has no comparison.",
    "AP by size uses COCOeval's default area ranges on the annotations' area field.",
]
METRICS = ["mAP_50_95", "mAP_50", "mAP_75", "AP_small", "AP_medium", "AP_large"]


def refuse_if_done(paths):
    done = [str(p) for p in paths if Path(p).exists()]
    if done:
        raise RuntimeError(f"refusing: test results already exist ({', '.join(done)}); the test set is used once")


def refuse_if_dirty(porcelain):
    if porcelain.strip():
        raise RuntimeError("refusing: uncommitted changes (commit the code first):\n" + porcelain)


def check_checkpoint(ck, metrics, ck_path):
    """The checkpoint must be the run's best epoch. Returns that epoch's history row."""
    problems = []
    if metrics.get("finished") is not True:
        problems.append("run not marked finished")
    for k_ck, k_m in [("epoch", "best_epoch"), ("seed", "seed"), ("git_commit", "git_commit")]:
        if ck.get(k_ck) != metrics.get(k_m):
            problems.append(f"{k_ck}: checkpoint {ck.get(k_ck)!r} vs metrics.json {metrics.get(k_m)!r}")
    if str(ck_path) != metrics.get("checkpoint"):
        problems.append(f"path {ck_path} vs metrics.json {metrics.get('checkpoint')}")
    rows = [r for r in metrics.get("history", []) if r.get("epoch") == metrics.get("best_epoch")]
    if len(rows) != 1:
        problems.append("best epoch not found exactly once in history")
    if problems:
        raise RuntimeError("checkpoint does not match metrics.json: " + "; ".join(problems))
    return rows[0]


def check_val(reproduced, recorded, tol=VAL_TOL):
    diff = abs(reproduced - recorded)
    if diff > tol:
        raise RuntimeError(f"validation not reproduced: {reproduced:.4f} vs recorded {recorded:.4f} "
                           f"(|diff| {diff:.4f} > {tol}); test set NOT touched")
    return diff


def _row(name, ours, paper):
    return {"metric": name, "ours": round(float(ours), 4), "paper": paper,
            "diff": None if paper is None else round(float(ours) - paper, 4)}


def comparison(test, paper=PAPER):
    rows = [_row(k, test[k], paper[k]) for k in METRICS]
    rows += [_row(f"AP {c}", test["per_class_AP"][c], p) for c, p in paper["per_class_AP"].items()]
    return rows


def main():
    import torch
    from torch.utils.data import DataLoader
    from vdv.data.det_data import COCO_ROOT, DetDataset, build_det_coco, collate
    from vdv.det_eval import coco_metrics
    from vdv.models.detector import build_detector
    from vdv.train_det import get_device, make_loaders, predict

    t0 = time.time()
    if not DRIVE.exists():
        raise RuntimeError("refusing: Drive not mounted, so the Drive copy of the results cannot be checked")
    refuse_if_done([RESULTS, DRIVE_RESULTS])
    git = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, check=True).stdout
    refuse_if_dirty(git("status", "--porcelain"))
    head = git("rev-parse", "HEAD").strip()
    if not torch.cuda.is_available():
        raise RuntimeError("refusing: no GPU (the validation check must use the same AMP inference as training)")

    metrics = json.loads(RUN_METRICS.read_text())
    ck_path = Path(metrics["checkpoint"])
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    best = check_checkpoint(ck, metrics, ck_path)
    dev = get_device()
    model = build_detector(pretrained=False)
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    print(f"checkpoint OK: epoch {ck['epoch']}, seed {ck['seed']}, commit {ck['git_commit'][:7]}")

    # 1. Validation reproduction: proves checkpoint, labels and evaluation before the one-off test run
    _, va_loader, va_json = make_loaders(SEED)
    val = coco_metrics(va_json, predict(model, va_loader, dev))
    recorded = best["val"]["mAP_50_95"]
    diff = check_val(val["mAP_50_95"], recorded)
    print(f"validation reproduced: {val['mAP_50_95']:.4f} vs recorded {recorded:.4f} (|diff| {diff:.4f} <= {VAL_TOL})")

    # 2. Test set, once
    te_json = build_det_coco("test", allow_test=True)      # /content/data/det_coco, outside the repo
    te = DetDataset(te_json, COCO_ROOT / "test2017", train=False)
    assert len(te) == N_TEST, f"expected {N_TEST} test images, got {len(te)}"
    loader = DataLoader(te, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate)
    dets = predict(model, loader, dev)
    test = coco_metrics(te_json, dets)

    for p in (DETS, DRIVE_DETS):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dets, default=float))
    rows = comparison(test)
    record = {
        "run": RUN, "split": "test", "n_test_images": len(te), "eval_git_commit": head,
        "checkpoint": str(ck_path), "checkpoint_epoch": ck["epoch"], "checkpoint_git_commit": ck["git_commit"],
        "val_reproduction": {"recorded": recorded, "reproduced": val["mAP_50_95"], "abs_diff": diff, "tolerance": VAL_TOL},
        "test": test, "paper": PAPER, "comparison": rows, "caveats": CAVEATS,
        "detections_saved_to": [str(DETS), str(DRIVE_DETS)],
        "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
        "seconds": round(time.time() - t0, 1), "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for p in (RESULTS, DRIVE_RESULTS):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(record, indent=2))

    print(f"\nTEST ({len(te)} images, {test['n_detections']} detections)   ours vs paper (Mask R-CNN R50, box AP)")
    print(f"{'metric':<20}{'ours':>8}{'paper':>8}{'diff':>8}")
    for r in rows:
        paper = "n/a" if r["paper"] is None else f"{r['paper']:.3f}"
        d = "" if r["diff"] is None else f"{r['diff']:+.3f}"
        print(f"{r['metric']:<20}{r['ours']:>8.3f}{paper:>8}{d:>8}")
    print("\nCaveats:\n- " + "\n- ".join(CAVEATS))
    print(f"\nsaved: {RESULTS.relative_to(REPO)} (+ Drive); detections: private folder + Drive only")


if __name__ == "__main__":
    main()
