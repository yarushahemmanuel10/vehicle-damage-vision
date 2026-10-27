"""A12 evaluation helpers (D-025): metrics, photo-level bootstrap, subgroups, run-once guard."""
from pathlib import Path

import numpy as np


def confusion(y, pred, k):
    return np.bincount(np.asarray(y) * k + np.asarray(pred), minlength=k * k).reshape(k, k)


def per_class(y, pred, k):
    """Per-class precision, recall, F1 and support (crop count), plus the confusion matrix."""
    cm = confusion(y, pred, k)
    tp, support, predicted = np.diag(cm).astype(float), cm.sum(axis=1), cm.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros(k), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(k), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(k), where=denom > 0)
    return precision, recall, f1, support, cm


def macro_f1(y, pred, k):
    """Macro F1 over the classes present in y (all six are present in the full val and test sets)."""
    _, _, f1, support, _ = per_class(y, pred, k)
    return float(f1[support > 0].mean())


def accuracy(y, pred):
    return float((np.asarray(y) == np.asarray(pred)).mean())


def bootstrap_by_photo(photo_ids, y, preds, k, n_resamples=2000, seed=42, reference=None, return_samples=False):
    """95% percentile CIs for accuracy and macro F1 of each model, resampling PHOTOS with replacement:
    all crops of a drawn photo come with it, because crops from one photo are not independent.
    With `reference`, also paired CIs of reference minus each other model on the same resamples."""
    y = np.asarray(y)
    photos = {}
    for i, p in enumerate(photo_ids):
        photos.setdefault(p, []).append(i)
    groups = [np.array(v) for v in photos.values()]
    rng = np.random.default_rng(seed)
    samples = {m: {"accuracy": [], "macro_f1": []} for m in preds}
    preds = {m: np.asarray(p) for m, p in preds.items()}
    for _ in range(n_resamples):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        for m, p in preds.items():
            samples[m]["accuracy"].append(accuracy(y[idx], p[idx]))
            samples[m]["macro_f1"].append(macro_f1(y[idx], p[idx], k))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out = {"n_resamples": n_resamples, "seed": seed, "n_photos": len(groups), "unit": "photo",
           "models": {m: {met: ci(v) for met, v in s.items()} for m, s in samples.items()}}
    if reference:
        out["paired_differences"] = {
            f"{reference} minus {m}": {met: ci(np.array(samples[reference][met]) - np.array(samples[m][met]))
                                       for met in ("accuracy", "macro_f1")}
            for m in preds if m != reference}
    if return_samples:
        out["samples"] = samples
    return out


def group_metrics(y, pred, groups, k, photo_ids=None):
    """Accuracy and macro F1 per subgroup, with crop (and photo) counts beside every figure."""
    y, pred, groups = np.asarray(y), np.asarray(pred), np.asarray(groups)
    out = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        row = {"n_crops": int(m.sum()), "accuracy": accuracy(y[m], pred[m]), "macro_f1": macro_f1(y[m], pred[m], k)}
        if photo_ids is not None:
            row["n_photos"] = int(len(set(np.asarray(photo_ids)[m].tolist())))
        out[str(g)] = row
    return out


def assert_not_evaluated(*paths):
    """The test set is evaluated once (A12): refuse if any results file already exists."""
    for p in paths:
        if Path(p).exists():
            raise RuntimeError(f"{p} exists: the test set has already been evaluated (run once, A12)")
            
