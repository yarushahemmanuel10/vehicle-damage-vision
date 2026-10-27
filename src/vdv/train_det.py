"""Phase B detection training (D-026). Never builds or reads the test split (B5 only).

    python -m vdv.train_det overfit                                   # 4 fixed train images: loss must fall below 30%
    python -m vdv.train_det smoke                                     # 32 train / 16 val images, 2 epochs, local only
    python -m vdv.train_det time                                      # one full train epoch + one val mAP pass, timed
    python -m vdv.train_det train --epochs N --val-every K [--warmup-iters W]   # real run (needs approval)
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from vdv.data.dataset import worker_init_fn
from vdv.data.det_data import COCO_ROOT, DET_DIR, DetDataset, build_det_coco, collate, filter_coco
from vdv.data.manifest import manifest_sha256
from vdv.det_eval import coco_metrics, to_coco_detections
from vdv.models.detector import MAX_SIZE, MIN_SIZE, build_detector
from vdv.seed import set_seed
from vdv.train_cls import CKPT_ROOT, DRIVE_RUNS, RUNS_DIR, TIMING_DIR, git_state, run_is_finished

RUN_PREFIX = "fasterrcnn_r50_fpn_v1"
BATCH, LR, MOMENTUM, WEIGHT_DECAY, NUM_WORKERS = 4, 0.005, 0.9, 1e-4, 2   # D-026
SMOKE_DIR = Path("/content/smoke_det")


def milestones(epochs):
    """D-026: lr x0.1 at about 2/3 and 11/12 of the epoch budget (24 epochs -> after epochs 16 and 22, as in the paper)."""
    return sorted({max(1, round(epochs * 2 / 3)), max(1, round(epochs * 11 / 12))})


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast(dev):
    return torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda")


def make_loaders(seed, n_train=None, n_val=None):
    tr_json, va_json = build_det_coco("train"), build_det_coco("val")
    train = DetDataset(tr_json, COCO_ROOT / "train2017", train=True, seed=seed)
    val = DetDataset(va_json, COCO_ROOT / "val2017", train=False)
    if n_val:
        names = [im["file_name"] for im in val.images[:n_val]]
        va_json = DET_DIR / "val_subset.json"
        va_json.write_text(json.dumps(filter_coco(json.loads((DET_DIR / "val.json").read_text()), names)))
        val = DetDataset(va_json, COCO_ROOT / "val2017", train=False)
    if n_train:
        train = Subset(train, range(min(n_train, len(train))))
    g = torch.Generator()
    g.manual_seed(seed)
    common = dict(batch_size=BATCH, num_workers=NUM_WORKERS, collate_fn=collate, worker_init_fn=worker_init_fn,
                  pin_memory=torch.cuda.is_available())
    return DataLoader(train, shuffle=True, generator=g, **common), DataLoader(val, shuffle=False, **common), va_json


def setup(epochs, dev, warmup_iters=0):
    model = build_detector(pretrained=True).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones(epochs), gamma=0.1)
    warm = (torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.001, total_iters=warmup_iters)
            if warmup_iters else None)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    return model, opt, sched, warm, scaler


def train_epoch(model, loader, opt, scaler, dev, warm=None):
    model.train()
    sums, n = {}, 0
    for imgs, targets, _ in loader:
        imgs = [i.to(dev, non_blocking=True) for i in imgs]
        targets = [{k: v.to(dev, non_blocking=True) for k, v in t.items()} for t in targets]
        with autocast(dev):
            losses = model(imgs, targets)
            loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss ({loss.item()}): stopping; check the learning rate or warmup")
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if warm is not None and warm.last_epoch < warm.total_iters:
            warm.step()
        for k, v in losses.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        sums["total"] = sums.get("total", 0.0) + float(loss)
        n += 1
    return {k: round(v / n, 5) for k, v in sums.items()}


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    dets = []
    for imgs, _, ids in loader:
        with autocast(dev):
            out = model([i.to(dev, non_blocking=True) for i in imgs])
        dets += to_coco_detections(out, ids)
    return dets


def settings(epochs, val_every, warmup_iters):
    return {"model": "torchvision fasterrcnn_resnet50_fpn (v1), COCO_V1 weights, boxes only",
            "min_size": MIN_SIZE, "max_size": MAX_SIZE,
            "augmentation": "train only: horizontal flip p=0.5, boxes flipped",
            "optimizer": "SGD", "lr": LR, "momentum": MOMENTUM, "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH, "epochs": epochs, "lr_milestones": milestones(epochs), "lr_gamma": 0.1,
            "warmup_iters": warmup_iters, "amp": True, "val_every": val_every,
            "selection": "best validation mAP@50:95; ties keep the earlier epoch; validation also at the last epoch",
            "torchvision_defaults": "3 trainable backbone stages, frozen BatchNorm in the backbone, "
                                    "box score threshold 0.05, 100 detections per image"}


def fit(epochs, seed, val_every, warmup_iters, out_dir, ckpt_dir, mirror_dir=None, n_train=None, n_val=None):
    set_seed(seed)
    dev = get_device()
    commit, dirty = git_state()
    tl, vl, gt = make_loaders(seed, n_train, n_val)
    model, opt, sched, warm, scaler = setup(epochs, dev, warmup_iters)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    record = {"run": out_dir.name, "seed": seed, "git_commit": commit, "git_dirty": dirty,
              "manifest_sha256": manifest_sha256(), "torch": torch.__version__,
              "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
              "n_train_images": len(tl.dataset), "n_val_images": len(vl.dataset),
              "settings": settings(epochs, val_every, warmup_iters), "history": [],
              "best_epoch": None, "best_val": None, "checkpoint": str(ckpt_dir / "best.pt"), "finished": False}

    def save():
        (out_dir / "metrics.json").write_text(json.dumps(record, indent=2))
        if mirror_dir:
            shutil.copytree(out_dir, mirror_dir, dirs_exist_ok=True)

    best = -1.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        lr_now = opt.param_groups[0]["lr"]
        losses = train_epoch(model, tl, opt, scaler, dev, warm if epoch == 1 else None)
        sched.step()
        row = {"epoch": epoch, "lr_at_start": lr_now, "train_losses": losses, "train_seconds": round(time.time() - t0, 1)}
        line = f"epoch {epoch}/{epochs}  lr {lr_now:.5f}  loss {losses['total']:.4f}  {row['train_seconds']}s"
        if epoch % val_every == 0 or epoch == epochs:
            t1 = time.time()
            m = coco_metrics(gt, predict(model, vl, dev))
            row["val"], row["val_seconds"] = m, round(time.time() - t1, 1)
            line += f"  val mAP50:95 {m['mAP_50_95']:.4f}  mAP50 {m['mAP_50']:.4f}  (val {row['val_seconds']}s)"
            if m["mAP_50_95"] > best:                                     # strict: ties keep the earlier epoch
                best = m["mAP_50_95"]
                torch.save({"model": model.state_dict(), "epoch": epoch, "seed": seed, "git_commit": commit},
                           ckpt_dir / "best.pt")
                record["best_epoch"], record["best_val"] = epoch, m
                line += "  *best*"
        record["history"].append(row)
        save()
        print(line, flush=True)
    record["finished"] = True
    save()
    return record


def overfit(seed, steps=150, n=4):
    """Train on 4 fixed train images (no flip). The total loss must fall below 30% of its start."""
    set_seed(seed)
    dev = get_device()
    ds = DetDataset(build_det_coco("train"), COCO_ROOT / "train2017", train=False)
    batch = [ds[i] for i in range(n)]
    imgs = [b[0].to(dev) for b in batch]
    targets = [{k: v.to(dev) for k, v in b[1].items()} for b in batch]
    model, opt, _, _, scaler = setup(1, dev)
    model.train()
    losses = []
    for step in range(steps + 1):
        with autocast(dev):
            loss = sum(model(imgs, targets).values())
        losses.append(float(loss))
        if step % 25 == 0:
            print(f"step {step:3d}  loss {float(loss):.4f}", flush=True)
        if step < steps:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
    ok = losses[-1] < 0.3 * losses[0]
    print(f"overfit {'PASS' if ok else 'FAIL'}: loss {losses[0]:.4f} -> {losses[-1]:.4f} (must fall below 30% of the start)")
    if not ok:
        sys.exit(1)


def time_epoch(seed):
    """One full train epoch + one validation pass (prediction + COCO evaluation), timed. No mAP is printed."""
    set_seed(seed)
    dev = get_device()
    commit, dirty = git_state()
    tl, vl, gt = make_loaders(seed)
    model, opt, sched, warm, scaler = setup(1, dev)
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    sync(); t0 = time.time()
    train_epoch(model, tl, opt, scaler, dev)
    sync(); t_train = time.time() - t0
    t0 = time.time()
    dets = predict(model, vl, dev)
    sync(); t_pred = time.time() - t0
    t0 = time.time()
    coco_metrics(gt, dets)
    t_eval = time.time() - t0
    rec = {"model": RUN_PREFIX, "seed": seed, "git_commit": commit, "git_dirty": dirty,
           "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
           "batch_size": BATCH, "min_size": MIN_SIZE, "max_size": MAX_SIZE, "amp": dev.type == "cuda",
           "train_images": len(tl.dataset), "train_iterations": len(tl), "val_images": len(vl.dataset),
           "train_epoch_seconds": round(t_train, 1), "val_predict_seconds": round(t_pred, 1),
           "val_coco_eval_seconds": round(t_eval, 1), "val_pass_seconds": round(t_pred + t_eval, 1),
           "note": "first epoch of a fresh process: includes CUDA start-up, so slightly pessimistic"}
    TIMING_DIR.mkdir(parents=True, exist_ok=True)
    (TIMING_DIR / f"{RUN_PREFIX}.json").write_text(json.dumps(rec, indent=2))
    print(f"{RUN_PREFIX}: train epoch {t_train:.0f}s ({len(tl)} iterations, {len(tl.dataset)} images) + "
          f"val pass {t_pred + t_eval:.0f}s (predict {t_pred:.0f}s + COCO eval {t_eval:.0f}s, {len(vl.dataset)} images) "
          f"on {rec['device']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["overfit", "smoke", "time", "train"])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--val-every", type=int)
    ap.add_argument("--warmup-iters", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.mode == "overfit":
        overfit(a.seed)
    elif a.mode == "smoke":
        run = f"{RUN_PREFIX}_seed{a.seed}_smoke"
        rec = fit(2, a.seed, 1, 0, SMOKE_DIR / run, SMOKE_DIR / run, n_train=32, n_val=16)
        assert rec["finished"] and Path(rec["checkpoint"]).exists() and all("val" in h for h in rec["history"])
        print(f"smoke PASS: 2 epochs, validation mAP computed each epoch, checkpoint in {SMOKE_DIR / run}")
    elif a.mode == "time":
        time_epoch(a.seed)
    else:
        if not a.epochs or not a.val_every:
            ap.error("--epochs and --val-every are required for a real run")
        _, dirty = git_state()
        if dirty:
            sys.exit("uncommitted changes: commit first, so the run is traceable to a commit")
        run = f"{RUN_PREFIX}_seed{a.seed}"
        if run_is_finished(RUNS_DIR / run, DRIVE_RUNS / run):
            sys.exit(f"{run} already finished: refusing to overwrite it")
        fit(a.epochs, a.seed, a.val_every, a.warmup_iters, RUNS_DIR / run, CKPT_ROOT / run, mirror_dir=DRIVE_RUNS / run)


if __name__ == "__main__":
    main()
