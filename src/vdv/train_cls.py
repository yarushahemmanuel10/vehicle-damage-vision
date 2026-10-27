"""A6 classification training (D-010, D-020, D-022). The test split is never loaded here (A12 only).

    python -m vdv.train_cls overfit --model resnet18_finetune   # one fixed batch; loss must fall below 10%
    python -m vdv.train_cls smoke   --model resnet18_finetune   # 256 train / 128 val crops, 2 epochs, local only
    python -m vdv.train_cls time    --model resnet18_finetune   # one full train epoch + one val pass, timed
    python -m vdv.train_cls train   --model resnet18_finetune --epochs N --seed 42   # real run (needs approval)
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset

from vdv.data.crops import crop_code_version
from vdv.data.dataset import make_dataset, worker_init_fn
from vdv.data.manifest import CLASSES, manifest_sha256
from vdv.models.classifiers import BACKBONE_LR, HEAD_LR, KINDS, build_model, set_train_mode
from vdv.seed import set_seed

REPO = Path(__file__).resolve().parents[2]
BATCH_SIZE, WEIGHT_DECAY, NUM_WORKERS = 64, 1e-4, 2
CKPT_ROOT = Path("/content/drive/MyDrive/vdv/checkpoints")
RUNS_DIR = REPO / "outputs/metrics/runs"
TIMING_DIR = REPO / "outputs/metrics/timing"
SMOKE_DIR = Path("/content/smoke")
DRIVE_RUNS = Path("/content/drive/MyDrive/vdv/runs")       # copy of every run folder; survives a runtime reset


def class_weights(labels, num_classes=len(CLASSES)):
    """D-022: inverse train class frequency, normalised to mean 1."""
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float64)
    if (counts == 0).any():
        raise ValueError(f"a class has no training examples: {counts.tolist()}")
    w = 1.0 / counts
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def git_state(repo=REPO):
    """(commit hash, dirty). Dirty means tracked files have uncommitted changes; untracked run outputs
    (timing, runs) do not count, but code must be committed."""
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True).stdout.strip()
    return run("rev-parse", "HEAD"), bool(run("status", "--porcelain", "--untracked-files=no"))


def run_is_finished(*run_dirs):
    """True if any copy of the run (local or Drive) has a metrics.json marked finished."""
    for d in run_dirs:
        f = Path(d) / "metrics.json"
        if f.exists() and json.loads(f.read_text()).get("finished"):
            return True
    return False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loaders(seed, n_train=None, n_val=None):
    train_full = make_dataset("train", augment=True, seed=seed)
    val_full = make_dataset("val", augment=False, seed=seed)
    train = Subset(train_full, range(min(n_train, len(train_full)))) if n_train else train_full
    val = Subset(val_full, range(min(n_val, len(val_full)))) if n_val else val_full
    g = torch.Generator()
    g.manual_seed(seed)
    common = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, worker_init_fn=worker_init_fn,
                  pin_memory=torch.cuda.is_available())
    return train_full, val_full, DataLoader(train, shuffle=True, generator=g, **common), \
        DataLoader(val, shuffle=False, **common)


def setup(kind, n_batches, epochs, train_labels, dev):
    model, head, groups = build_model(kind)
    model.to(dev)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_labels).to(dev))
    opt = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * n_batches))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    return model, head, criterion, opt, sched, scaler


def autocast(dev):
    return torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda")


def train_epoch(model, head, kind, loader, criterion, opt, sched, scaler, dev):
    set_train_mode(model, head, kind)
    total, n = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with autocast(dev):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()                                                  # cosine, stepped every batch
        total += loss.item() * len(y)
        n += len(y)
    return total / n


@torch.no_grad()
def evaluate(model, loader, criterion, dev):
    model.eval()
    probs, labels, idx, total, n = [], [], [], 0.0, 0
    for x, y, i in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with autocast(dev):
            logits = model(x)
        logits = logits.float()
        total += criterion(logits, y).item() * len(y)
        n += len(y)
        probs.append(torch.softmax(logits, 1).cpu()); labels.append(y.cpu()); idx.append(i)
    p, yv, iv = torch.cat(probs).numpy(), torch.cat(labels).numpy(), torch.cat(idx).numpy()
    pred = p.argmax(1)
    metrics = {"loss": total / n, "accuracy": float(accuracy_score(yv, pred)),
               "macro_f1": float(f1_score(yv, pred, average="macro", labels=list(range(len(CLASSES))),
                                          zero_division=0))}
    return metrics, p, yv, iv


def write_val_probs(path, rows, p, yv, iv):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "ann_id", "label", "pred", *[f"p_{c.replace(' ', '_')}" for c in CLASSES]])
        for k in range(len(iv)):
            r = rows[int(iv[k])]
            w.writerow([r["split"], r["ann_id"], int(yv[k]), int(p[k].argmax()), *[f"{v:.6f}" for v in p[k]]])


def settings(kind, epochs, train_labels):
    return {"loss": "class-weighted cross-entropy, weights = inverse train frequency normalised to mean 1",
            "class_weights": dict(zip(CLASSES, [round(float(w), 4) for w in class_weights(train_labels)])),
            "optimizer": "AdamW", "weight_decay": WEIGHT_DECAY,
            "lr_head": HEAD_LR,
            "lr_backbone": BACKBONE_LR if kind == "resnet18_finetune" else ("frozen" if "frozen" in kind or "probe" in kind else HEAD_LR),
            "schedule": "cosine decay to 0 over all steps, stepped per batch, no warmup",
            "batch_size": BATCH_SIZE, "amp": True, "epochs": epochs,
            "selection": "best validation macro F1; ties go to the earlier epoch",
            "augmentation": "train only: horizontal flip p=0.5, brightness x Uniform[0.8, 1.2] (D-021)"}


def fit(kind, epochs, seed, out_dir, ckpt_dir, n_train=None, n_val=None, mirror_dir=None):
    set_seed(seed)
    dev = get_device()
    commit, dirty = git_state()
    train_full, val_full, tl, vl = make_loaders(seed, n_train, n_val)
    model, head, criterion, opt, sched, scaler = setup(kind, len(tl), epochs, train_full.labels, dev)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    record = {"run": out_dir.name, "model": kind, "seed": seed, "git_commit": commit, "git_dirty": dirty,
              "crop_version": crop_code_version(), "manifest_sha256": manifest_sha256(),
              "torch": torch.__version__,
              "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
              "n_train": len(tl.dataset), "n_val": len(vl.dataset),
              "settings": settings(kind, epochs, train_full.labels),
              "history": [], "best_epoch": None, "best_val_macro_f1": None,
              "checkpoint": str(ckpt_dir / "best.pt"), "finished": False}
    best = -1.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, head, kind, tl, criterion, opt, sched, scaler, dev)
        m, p, yv, iv = evaluate(model, vl, criterion, dev)
        record["history"].append({"epoch": epoch, "train_loss": round(train_loss, 5),
                                  "val_loss": round(m["loss"], 5), "val_accuracy": round(m["accuracy"], 5),
                                  "val_macro_f1": round(m["macro_f1"], 5), "seconds": round(time.time() - t0, 1)})
        if m["macro_f1"] > best:                                      # strict: ties keep the earlier epoch
            best = m["macro_f1"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "kind": kind, "seed": seed,
                        "git_commit": commit}, ckpt_dir / "best.pt")
            write_val_probs(out_dir / "val_probs.csv", val_full.rows, p, yv, iv)
            record["best_epoch"], record["best_val_macro_f1"] = epoch, round(best, 5)
        (out_dir / "metrics.json").write_text(json.dumps(record, indent=2))
        if mirror_dir:
            shutil.copytree(out_dir, mirror_dir, dirs_exist_ok=True)
        h = record["history"][-1]
        print(f"epoch {epoch}/{epochs}  train_loss {h['train_loss']:.4f}  val_loss {h['val_loss']:.4f}  "
              f"val_acc {h['val_accuracy']:.4f}  val_macroF1 {h['val_macro_f1']:.4f}  {h['seconds']}s"
              + ("  *best*" if record["best_epoch"] == epoch else ""), flush=True)
    record["finished"] = True
    (out_dir / "metrics.json").write_text(json.dumps(record, indent=2))
    if mirror_dir:
        shutil.copytree(out_dir, mirror_dir, dirs_exist_ok=True)
    return record


def overfit(kind, seed, steps=150, n=16):
    """Train on one fixed batch of n real train crops (no augmentation). The loss must fall below 10%."""
    set_seed(seed)
    dev = get_device()
    ds = make_dataset("train", augment=False, seed=seed)
    picks = np.linspace(0, len(ds) - 1, n).astype(int)
    x = torch.stack([ds[int(i)][0] for i in picks]).to(dev)
    y = torch.tensor([ds[int(i)][1] for i in picks]).to(dev)
    model, head, criterion, opt, _, scaler = setup(kind, 1, 1, ds.labels, dev)
    losses = []
    for step in range(steps + 1):
        set_train_mode(model, head, kind)
        opt.zero_grad(set_to_none=True)
        with autocast(dev):
            loss = criterion(model(x), y)
        losses.append(loss.item())
        if step % 25 == 0:
            print(f"step {step:3d}  loss {loss.item():.4f}", flush=True)
        if step < steps:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
    ok = losses[-1] < 0.1 * losses[0]
    print(f"overfit {'PASS' if ok else 'FAIL'}: loss {losses[0]:.4f} -> {losses[-1]:.4f} "
          f"(must fall below 10% of the start)")
    if not ok:
        sys.exit(1)


def time_epoch(kind, seed):
    """One full train epoch + one val pass, timed. No validation scores are printed or saved."""
    set_seed(seed)
    dev = get_device()
    commit, dirty = git_state()
    train_full, val_full, tl, vl = make_loaders(seed)
    model, head, criterion, opt, sched, scaler = setup(kind, len(tl), 1, train_full.labels, dev)
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    sync(); t0 = time.time()
    train_epoch(model, head, kind, tl, criterion, opt, sched, scaler, dev)
    sync(); t_train = time.time() - t0
    t0 = time.time()
    evaluate(model, vl, criterion, dev)
    sync(); t_val = time.time() - t0
    rec = {"model": kind, "seed": seed, "git_commit": commit, "git_dirty": dirty,
           "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
           "batch_size": BATCH_SIZE, "num_workers": NUM_WORKERS, "amp": dev.type == "cuda",
           "train_batches": len(tl), "val_batches": len(vl),
           "train_epoch_seconds": round(t_train, 1), "val_pass_seconds": round(t_val, 1),
           "epoch_total_seconds": round(t_train + t_val, 1),
           "note": "first epoch of a fresh process: includes CUDA start-up, so slightly pessimistic"}
    TIMING_DIR.mkdir(parents=True, exist_ok=True)
    (TIMING_DIR / f"{kind}.json").write_text(json.dumps(rec, indent=2))
    print(f"{kind}: train epoch {t_train:.0f}s + val pass {t_val:.0f}s = {t_train + t_val:.0f}s per epoch "
          f"on {rec['device']} ({len(tl)} train batches, {len(vl)} val batches)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["overfit", "smoke", "time", "train"])
    ap.add_argument("--model", choices=KINDS, required=True)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.mode == "overfit":
        overfit(a.model, a.seed)
    elif a.mode == "smoke":
        run = f"{a.model}_seed{a.seed}_smoke"
        rec = fit(a.model, 2, a.seed, SMOKE_DIR / run, SMOKE_DIR / run, n_train=256, n_val=128)
        with open(SMOKE_DIR / run / "val_probs.csv") as f:
            n_probs = sum(1 for _ in f) - 1
        assert len(rec["history"]) == 2 and Path(rec["checkpoint"]).exists() and n_probs == 128
        print(f"smoke PASS: 2 epochs, checkpoint and {n_probs} val probabilities written to {SMOKE_DIR / run}")
    elif a.mode == "time":
        time_epoch(a.model, a.seed)
    else:
        if not a.epochs:
            ap.error("--epochs is required for a real run")
        _, dirty = git_state()
        if dirty:
            sys.exit("uncommitted changes: commit first, so every run is traceable to a commit")
        run = f"{a.model}_seed{a.seed}"
        if run_is_finished(RUNS_DIR / run, DRIVE_RUNS / run):
            sys.exit(f"{run} already finished: refusing to overwrite it")
        fit(a.model, a.epochs, a.seed, RUNS_DIR / run, CKPT_ROOT / run, mirror_dir=DRIVE_RUNS / run)


if __name__ == "__main__":
    main()
