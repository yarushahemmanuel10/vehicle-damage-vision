"""A11: review-threshold table, calibration and reliability on VALIDATION for fine-tune seed 42 (D-024).

Usage: python scripts/review_threshold.py
Writes outputs/metrics/review_threshold_val.json and outputs/figures/reliability_val_finetune_seed42.png
(a chart only: no dataset images)."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vdv.data.manifest import CLASSES
from vdv.review import reliability, select_threshold, threshold_table

RUN = "resnet18_finetune_seed42"                                       # D-024
CUTOFFS = [0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 0.995]            # D-024
MIN_AUTO_ACCURACY = 0.97                                               # D-024 selection rule
N_BINS = 10
REPO = Path(__file__).resolve().parents[1]


def main():
    with open(REPO / f"outputs/metrics/runs/{RUN}/val_probs.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows and all(r["split"] == "val" for r in rows), "validation probabilities only"
    cols = [f"p_{c.replace(' ', '_')}" for c in CLASSES]
    probs = np.array([[float(r[c]) for c in cols] for r in rows])
    y = np.array([int(r["label"]) for r in rows])

    table = threshold_table(y, probs, CUTOFFS)
    chosen, met = select_threshold(table, MIN_AUTO_ACCURACY)
    ece, bins = reliability(y, probs, N_BINS)
    result = {"split": "val", "run": RUN, "n": int(len(y)),
              "overall_accuracy": float((probs.argmax(1) == y).mean()),
              "cutoffs": CUTOFFS, "table": table,
              "selection_rule": f"lowest cut-off with auto-labelled accuracy >= {MIN_AUTO_ACCURACY}; "
                                "if none, the highest cut-off",
              "operating_threshold": chosen, "target_met": met,
              "ece": round(ece, 4), "ece_bins": N_BINS, "reliability": bins}
    (REPO / "outputs/metrics/review_threshold_val.json").write_text(json.dumps(result, indent=2))

    used = [b for b in bins if b["n"]]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    ax.plot([b["mean_confidence"] for b in used], [b["accuracy"] for b in used], "o-",
            label=f"fine-tune seed 42 (ECE {ece:.3f})")
    ax.set_xlabel("mean confidence (top softmax probability)")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Reliability on validation (n={len(y)})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(REPO / "outputs/figures/reliability_val_finetune_seed42.png", dpi=120)

    print(f"{RUN} on validation: n={len(y)}, overall accuracy {result['overall_accuracy']:.4f}\n")
    print(f"{'cut-off':>7s} {'to review':>10s} {'share':>7s} {'auto acc':>9s} {'auto macroF1':>13s} {'classes':>8s}")
    for r in table:
        acc = "-" if r["auto_accuracy"] is None else f"{r['auto_accuracy']:.4f}"
        mf1 = "-" if r["auto_macro_f1"] is None else f"{r['auto_macro_f1']:.4f}"
        print(f"{r['cutoff']:7.3f} {r['n_review']:10d} {r['share_review']:7.1%} {acc:>9s} {mf1:>13s} {r['classes_in_auto']:8d}")
    print(f"\noperating threshold (D-024 rule): {chosen}" + ("" if met else
          f"  (NO cut-off reached {MIN_AUTO_ACCURACY:.0%}: highest cut-off used)"))
    print(f"\nexpected calibration error: {ece:.4f}  ({N_BINS} equal-width bins)")
    print(f"{'bin':>12s} {'n':>5s} {'mean conf':>10s} {'accuracy':>9s} {'gap':>8s}")
    for b in used:
        print(f"{b['bin']:>12s} {b['n']:5d} {b['mean_confidence']:10.4f} {b['accuracy']:9.4f} {b['gap']:8.4f}")


if __name__ == "__main__":
    main()
