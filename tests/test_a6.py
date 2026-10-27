import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset

from vdv.data.dataset import IMAGENET_MEAN, IMAGENET_STD, CropDataset, make_dataset, to_tensor, worker_init_fn
from vdv.models.classifiers import BACKBONE_LR, HEAD_LR, build_model, set_train_mode
from vdv.train_cls import class_weights


def test_pure_red_pixel_ends_up_in_channel_0():
    bgr = np.zeros((4, 4, 3), np.uint8)
    bgr[..., 2] = 255                                   # pure red in OpenCV's BGR order
    x = to_tensor(bgr)
    assert x.shape == (3, 4, 4)
    red = float((1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0])
    assert torch.allclose(x[0], torch.full((4, 4), red))
    assert (x[1] < 0).all() and (x[2] < 0).all()


def test_test_split_is_locked():
    with pytest.raises(RuntimeError):
        make_dataset("test", augment=False)


def fake_dataset(tmp_path, n=8):
    rows = []
    for i in range(n):
        cv2.imwrite(str(tmp_path / f"{i}.png"), np.full((50, 100, 3), 120, np.uint8))
        rows.append({"crop_path": f"{i}.png", "cls": "dent"})
    return CropDataset(rows, tmp_path, augment=True, seed=42, return_aug=True)


def brightness_per_worker(tmp_path, init_fn):
    g = torch.Generator()
    g.manual_seed(42)
    dl = DataLoader(Subset(fake_dataset(tmp_path), range(8)), batch_size=1, num_workers=2,
                    worker_init_fn=init_fn, generator=g)
    factors = [float(b[3]) for b in dl]                 # batch_size 1: items alternate worker 0 / worker 1
    return factors[0::2], factors[1::2]


def test_workers_get_different_augmentations(tmp_path):
    w0, w1 = brightness_per_worker(tmp_path, worker_init_fn)
    assert w0 != w1


def test_without_worker_seeding_workers_repeat_each_other(tmp_path):
    w0, w1 = brightness_per_worker(tmp_path, None)      # why worker_init_fn is needed
    assert w0 == w1


def test_class_weights_inverse_frequency_mean_one():
    w = class_weights([0, 0, 0, 1], num_classes=2)      # counts 3 and 1
    assert torch.allclose(w, torch.tensor([0.5, 1.5]))
    assert torch.isclose(w.mean(), torch.tensor(1.0))


def trainable(model):
    return {n for n, p in model.named_parameters() if p.requires_grad}


def test_resnet_frozen_trains_head_only():
    model, _, groups = build_model("resnet18_frozen", pretrained=False)
    assert trainable(model) == {"fc.weight", "fc.bias"}
    assert [g["lr"] for g in groups] == [HEAD_LR]


def test_resnet_finetune_has_lower_backbone_lr():
    model, _, groups = build_model("resnet18_finetune", pretrained=False)
    assert len(trainable(model)) == len(list(model.parameters()))
    assert [g["lr"] for g in groups] == [HEAD_LR, BACKBONE_LR]


def test_resnet_scratch_trains_everything_at_one_lr():
    model, _, groups = build_model("resnet18_scratch", pretrained=False)
    assert len(trainable(model)) == len(list(model.parameters()))
    assert [g["lr"] for g in groups] == [HEAD_LR]


def test_vit_probe_trains_head_only_and_outputs_six_classes():
    model, head, groups = build_model("vit_b16_probe", pretrained=False)
    assert trainable(model) == {"heads.head.weight", "heads.head.bias"}
    model.eval()
    with torch.no_grad():
        assert model(torch.zeros(1, 3, 224, 224)).shape == (1, 6)


def test_frozen_backbone_stays_in_eval_mode():
    model, head, _ = build_model("resnet18_frozen", pretrained=False)
    set_train_mode(model, head, "resnet18_frozen")
    assert not model.bn1.training and head.training


import json
import subprocess

from vdv.train_cls import git_state, run_is_finished


def test_untracked_files_do_not_make_repo_dirty(tmp_path):
    g = lambda *a: subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)
    g("init", "-q")
    (tmp_path / "code.py").write_text("x = 1\n")
    g("add", "code.py")
    g("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    (tmp_path / "metrics.json").write_text("{}")         # an untracked run output
    assert git_state(tmp_path)[1] is False
    (tmp_path / "code.py").write_text("x = 2\n")        # an uncommitted code change
    assert git_state(tmp_path)[1] is True


def test_only_finished_runs_are_refused(tmp_path):
    local, drive = tmp_path / "local", tmp_path / "drive"
    assert not run_is_finished(local, drive)                                  # no run yet
    local.mkdir()
    (local / "metrics.json").write_text(json.dumps({"finished": False}))
    assert not run_is_finished(local, drive)                                  # interrupted: may restart
    drive.mkdir()
    (drive / "metrics.json").write_text(json.dumps({"finished": True}))
    assert run_is_finished(local, drive)                                      # finished on Drive: refused
