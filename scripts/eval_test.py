"""A12: the one-off test-set evaluation (D-025). Refuses to run twice or on uncommitted code.

Usage: python scripts/eval_test.py
Reads the 12 best checkpoints from Drive. Writes (in git) outputs/metrics/runs/<run>/test_probs.csv,
outputs/metrics/test_results.json and outputs/metrics/a13_examples.json; (git-ignored)
outputs/figures_private/a13_examples_test.png. Copies the results to Drive.
"""
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from vdv.data.crops import crop_code_version, pad_to_square
from vdv.data.dataset import DEFAULT_CROP_DIR, make_dataset
from vdv.data.image_info import load_image_info
from vdv.data.manifest import CLASSES, load_split, manifest_sha256
from vdv.evaluation import accuracy, assert_not_evaluated, bootstrap_by_photo, group_metrics, macro_f1, per_class
from vdv.models.classifiers import build_model
from vdv.review import threshold_table
from vdv.train_cls import CKPT_ROOT, DRIVE_RUNS, RUNS_DIR, git_state, write_val_probs

REPO = Path(__file__).resolve().parents[1]
MODELS = ["resnet18_scratch", "resnet18_frozen", "resnet18_finetune", "vit_b16_probe"]
SEEDS = [42, 43, 44]
REFERENCE = "resnet18_finetune"                                   # D-024 / D-025
K = len(CLASSES)
RESULTS = REPO / "outputs/metrics/test_results.json"
EXAMPLES = REPO / "outputs/metrics/a13_examples.json"
FIGURE = REPO / "outputs/figures_private/a13_examples_test.png"
DRIVE_TEST = Path("/content/drive/MyDrive/vdv/test")
INFO_XLSX = Path("/content/data/CarDD_release/CarDD_COCO/annotations/image_info.xlsx")


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    probs, labels, idx = [], [], []
    for x, y, i in loader:
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            logits = model(x.to(dev, non_blocking=True))
        probs.append(torch.softmax(logits.float(), 1).cpu()); labels.append(y); idx.append(i)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy(), torch.cat(idx).numpy()


def run_metrics(y, p):
    pred = p.argmax(1)
    precision, recall, f1, support, cm = per_class(y, pred, K)
    return {"accuracy": accuracy(y, pred), "macro_f1": macro_f1(y, pred, K),
            "per_class": {c: {"precision": float(precision[j]), "recall": float(recall[j]),
                              "f1": float(f1[j]), "n_crops": int(support[j])} for j, c in enumerate(CLASSES)},
            "confusion_matrix": cm.tolist()}


def subgroups(rows, y, pred, info):
    """D-019 clean vs contaminated; D-012 colour and angle bins; D-012 completeness as counts only."""
    files = [r["file_name"] for r in rows]
    contam = ["contaminated (other_class_ge50=1)" if r["other_class_ge50"] == "1" else "clean (other_class_ge50=0)"
              for r in rows]
    colour = [info.loc[f, "colour_bin"] for f in files]
    angle = [info.loc[f, "angle_bin"] for f in files]
    complete = np.array([info.loc[f, "completeness"] for f in files])
    return {"D-019 contamination": group_metrics(y, pred, contam, K, files),
            "D-012 colour": group_metrics(y, pred, colour, K, files),
            "D-012 angle": group_metrics(y, pred, angle, K, files),
            "D-012 completeness (counts only)": {
                g: {"n_crops": int((complete == g).sum()),
                    "n_photos": int(len({f for f, c in zip(files, complete) if c == g}))}
                for g in sorted(set(complete.tolist()))}}


def pick_examples(y, p, threshold):
    """A13 (D-025): highest-confidence correct crop per class; 4 most confident errors;
    2 lowest-confidence crops below the operating threshold."""
    conf, pred = p.max(1), p.argmax(1)
    correct, chosen = pred == y, []
    for j in range(K):
        cand = np.flatnonzero(correct & (y == j))
        if len(cand):
            chosen.append(("correct", int(cand[np.argmax(conf[cand])])))
    wrong = np.flatnonzero(~correct)
    chosen += [("error", int(i)) for i in wrong[np.argsort(-conf[wrong], kind="stable")][:4]]
    low = [int(i) for i in np.flatnonzero(conf < threshold)[np.argsort(conf[conf < threshold], kind="stable")]]
    taken = {i for _, i in chosen}
    chosen += [("review", i) for i in [i for i in low if i not in taken][:2]]
    return chosen


def figure(rows, y, p, chosen, threshold):
    tiles = []
    for n, (kind, i) in enumerate(chosen, 1):
        img = pad_to_square(cv2.imread(str(DEFAULT_CROP_DIR / rows[i]["crop_path"])))
        strip = np.full((40, 224, 3), 255, np.uint8)
        conf, pred = float(p[i].max()), int(p[i].argmax())
        route = "review" if conf < threshold else "auto"
        cv2.putText(strip, f"#{n} {kind}: true {CLASSES[int(y[i])]}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, 0, 1, cv2.LINE_AA)
        cv2.putText(strip, f"pred {CLASSES[pred]} {conf:.3f} {route}", (4, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38, 0, 1, cv2.LINE_AA)
        tiles.append(np.vstack([img, strip]))
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 255))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(FIGURE), np.vstack([np.hstack(tiles[r:r + 4]) for r in range(0, len(tiles), 4)]))


def main():
    assert_not_evaluated(RESULTS, DRIVE_TEST / "test_results.json")
    commit, dirty = git_state()
    if dirty:
        sys.exit("uncommitted changes to tracked files: commit first, so the test results are traceable")
    threshold = json.loads((REPO / "outputs/metrics/review_threshold_val.json").read_text())["operating_threshold"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = make_dataset("test", augment=False, allow_test=True)               # A12: the one place test is read
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=dev.type == "cuda")
    y = np.array(ds.labels)
    files = [r["file_name"] for r in ds.rows]

    runs, probs = {}, {}
    for m in MODELS:
        for s in SEEDS:
            run = f"{m}_seed{s}"
            rec = json.loads((RUNS_DIR / run / "metrics.json").read_text())
            ckpt = torch.load(CKPT_ROOT / run / "best.pt", map_location="cpu")
            assert rec["finished"] and ckpt["epoch"] == rec["best_epoch"] and ckpt["kind"] == m and ckpt["seed"] == s, \
                f"{run}: checkpoint does not match metrics.json"
            model, _, _ = build_model(m, pretrained=False)
            model.load_state_dict(ckpt["model"])
            model.to(dev)
            p, yy, iv = predict(model, loader, dev)
            assert (iv == np.arange(len(ds))).all() and (yy == y).all()
            write_val_probs(RUNS_DIR / run / "test_probs.csv", ds.rows, p, yy, iv)
            runs[run] = {"model": m, "seed": s, "best_epoch": rec["best_epoch"],
                         "checkpoint_commit": ckpt["git_commit"], **run_metrics(y, p)}
            probs[run] = p
            print(f"evaluated {run}", flush=True)
            del model
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    per_model = {}
    for m in MODELS:
        per_model[m] = {}
        for met in ("accuracy", "macro_f1"):
            v = [runs[f"{m}_seed{s}"][met] for s in SEEDS]
            per_model[m][met] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "per_seed": v}
    boot = bootstrap_by_photo(files, y, {m: probs[f"{m}_seed42"].argmax(1) for m in MODELS}, K,
                              n_resamples=2000, seed=42, reference=REFERENCE)

    ref = probs[f"{REFERENCE}_seed42"]
    review = threshold_table(y, ref, [threshold])[0]
    info = load_image_info(INFO_XLSX).set_index("file_name")
    sub_test = subgroups(ds.rows, y, ref.argmax(1), info)

    with open(RUNS_DIR / f"{REFERENCE}_seed42" / "val_probs.csv", newline="") as f:
        vrows = list(csv.DictReader(f))
    man_val = {r["ann_id"]: r for r in load_split("val")}
    sub_val = subgroups([man_val[r["ann_id"]] for r in vrows], np.array([int(r["label"]) for r in vrows]),
                        np.array([int(r["pred"]) for r in vrows]), info)

    chosen = pick_examples(y, ref, threshold)
    figure(ds.rows, y, ref, chosen, threshold)
    examples = [{"number": n, "kind": kind, "split": "test", "ann_id": ds.rows[i]["ann_id"],
                 "file_name": ds.rows[i]["file_name"], "true": CLASSES[int(y[i])],
                 "pred": CLASSES[int(ref[i].argmax())], "confidence": round(float(ref[i].max()), 4),
                 "routed": "review" if ref[i].max() < threshold else "auto"}
                for n, (kind, i) in enumerate(chosen, 1)]

    results = {"split": "test", "git_commit": commit, "git_dirty": dirty, "crop_version": crop_code_version(),
               "manifest_sha256": manifest_sha256(), "n_crops": int(len(y)), "n_photos": int(len(set(files))),
               "runs": runs, "per_model_over_seeds": per_model, "bootstrap_seed42": boot,
               "review_at_operating_threshold": {"model": f"{REFERENCE}_seed42", "threshold": threshold, **review},
               "subgroups_finetune_seed42": {"test": sub_test, "val": sub_val}}
    RESULTS.write_text(json.dumps(results, indent=2))
    EXAMPLES.write_text(json.dumps(examples, indent=2))
    DRIVE_TEST.mkdir(parents=True, exist_ok=True)
    for f in (RESULTS, EXAMPLES):
        shutil.copy(f, DRIVE_TEST / f.name)
    for run in runs:
        (DRIVE_RUNS / run).mkdir(parents=True, exist_ok=True)
        shutil.copy(RUNS_DIR / run / "test_probs.csv", DRIVE_RUNS / run / "test_probs.csv")

    # --- Summary ---
    print(f"\nTEST: {len(y)} crops from {len(set(files))} photos | commit {commit[:7]}")
    print(f"\n{'model':20s} {'acc mean±SD':>16s} {'macroF1 mean±SD':>18s}   seed 42: acc [95% CI]        macroF1 [95% CI]")
    for m in MODELS:
        a, f1 = per_model[m]["accuracy"], per_model[m]["macro_f1"]
        r42, b = runs[f"{m}_seed42"], boot["models"][m]
        print(f"{m:20s} {a['mean']:.4f}±{a['sd']:.4f}  {f1['mean']:.4f}±{f1['sd']:.4f}     "
              f"{r42['accuracy']:.4f} [{b['accuracy'][0]:.4f}, {b['accuracy'][1]:.4f}]  "
              f"{r42['macro_f1']:.4f} [{b['macro_f1'][0]:.4f}, {b['macro_f1'][1]:.4f}]")
    print("\npaired differences (seed 42, same photo resamples), 95% CI:")
    for k, v in boot["paired_differences"].items():
        print(f"  {k:42s} accuracy [{v['accuracy'][0]:+.4f}, {v['accuracy'][1]:+.4f}]  "
              f"macro F1 [{v['macro_f1'][0]:+.4f}, {v['macro_f1'][1]:+.4f}]")
    print(f"\nper class, {REFERENCE}_seed42:")
    for c, v in runs[f"{REFERENCE}_seed42"]["per_class"].items():
        print(f"  {c:14s} precision {v['precision']:.3f}  recall {v['recall']:.3f}  F1 {v['f1']:.3f}  (n={v['n_crops']})")
    print(f"\nreview at threshold {threshold}: {review['n_review']} of {len(y)} crops to review "
          f"({review['share_review']:.1%}); auto-labelled accuracy {review['auto_accuracy']:.4f}, "
          f"macro F1 {review['auto_macro_f1']:.4f} ({review['classes_in_auto']} classes)")
    for split, sub in (("test", sub_test), ("val", sub_val)):
        print(f"\nsubgroups ({split}, {REFERENCE}_seed42):")
        for name, groups in sub.items():
            for g, v in groups.items():
                extra = f"acc {v['accuracy']:.4f}  macroF1 {v['macro_f1']:.4f}" if "accuracy" in v else ""
                print(f"  {name:34s} {g:36s} n_crops {v['n_crops']:4d}  n_photos {v['n_photos']:4d}  {extra}")
    print("\nA13 examples:")
    for e in examples:
        print(f"  #{e['number']:2d} {e['kind']:8s} ann {e['ann_id']:>5s}  true {e['true']:14s} pred {e['pred']:14s} "
              f"conf {e['confidence']:.4f}  {e['routed']}")
    print("\nwritten:", RESULTS, "|", EXAMPLES, "|", FIGURE, "| copies on Drive")


if __name__ == "__main__":
    main()
