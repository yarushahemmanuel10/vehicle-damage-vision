"""A11 human-in-the-loop routing (D-024). Confidence = top softmax probability.
A crop with confidence BELOW the cut-off goes to human review; at or above it, it is auto-labelled."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def threshold_table(y, probs, cutoffs):
    y, probs = np.asarray(y), np.asarray(probs, float)
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    rows = []
    for t in cutoffs:
        auto = conf >= t
        row = {"cutoff": t, "n_auto": int(auto.sum()), "n_review": int((~auto).sum()),
               "share_review": float((~auto).mean())}
        if auto.any():
            row["auto_accuracy"] = float(accuracy_score(y[auto], pred[auto]))
            row["auto_macro_f1"] = float(f1_score(y[auto], pred[auto], average="macro", zero_division=0))
            row["classes_in_auto"] = int(len(set(y[auto].tolist())))
        else:
            row.update(auto_accuracy=None, auto_macro_f1=None, classes_in_auto=0)
        rows.append(row)
    return rows


def select_threshold(table, min_accuracy):
    """D-024 rule: the lowest cut-off whose auto-labelled accuracy is >= min_accuracy.
    If none reaches it: the highest cut-off. Returns (cut-off, target_met)."""
    ok = [r for r in table if r["auto_accuracy"] is not None and r["auto_accuracy"] >= min_accuracy]
    if ok:
        return min(ok, key=lambda r: r["cutoff"])["cutoff"], True
    return max(r["cutoff"] for r in table), False


def reliability(y, probs, n_bins=10):
    """Expected calibration error and a reliability table over equal-width confidence bins.
    ECE = sum over bins of (share of crops in bin) x |accuracy - mean confidence|."""
    y, probs = np.asarray(y), np.asarray(probs, float)
    conf, correct = probs.max(axis=1), probs.argmax(axis=1) == y
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, bins = 0.0, []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = ((conf >= lo) if i == 0 else (conf > lo)) & (conf <= hi)
        n = int(m.sum())
        row = {"bin": f"{'[' if i == 0 else '('}{lo:.1f}, {hi:.1f}]", "n": n,
               "mean_confidence": None, "accuracy": None, "gap": None}
        if n:
            c, a = float(conf[m].mean()), float(correct[m].mean())
            ece += n / len(y) * abs(a - c)
            row.update(mean_confidence=round(c, 4), accuracy=round(a, 4), gap=round(a - c, 4))
        bins.append(row)
    return float(ece), bins
