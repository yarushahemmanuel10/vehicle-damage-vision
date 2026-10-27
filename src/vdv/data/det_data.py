"""Phase B detection data (D-026, D-027). COCO files are built at runtime from the frozen split
(duplicate-excluded photos removed) on Colab's disk and are never committed (CarDD licence)."""
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from vdv.data.manifest import load_split

cv2.setNumThreads(0)
COCO_ROOT = Path("/content/data/CarDD_release/CarDD_COCO")
DET_DIR = Path("/content/data/det_coco")                    # runtime only, outside the repo


def filter_coco(coco, keep_file_names):
    """Copy of a COCO dict with only the listed images and their annotations."""
    keep = set(keep_file_names)
    images = [im for im in coco["images"] if im["file_name"] in keep]
    ids = {im["id"] for im in images}
    return {"images": copy.deepcopy(images),
            "annotations": copy.deepcopy([a for a in coco["annotations"] if a["image_id"] in ids]),
            "categories": copy.deepcopy(coco["categories"])}


def build_det_coco(split, root=COCO_ROOT, out_dir=DET_DIR, allow_test=False):
    """Write <out_dir>/<split>.json: the official annotations restricted to the frozen split's photos."""
    if split == "test" and not allow_test:
        raise RuntimeError("The test split is locked until the final evaluation (B5)")
    coco = json.loads((Path(root) / "annotations" / f"instances_{split}2017.json").read_text())
    keep = {r["file_name"] for r in load_split(split)}       # verified against the frozen manifest hash
    out = filter_coco(coco, keep)
    assert {im["file_name"] for im in out["images"]} == keep, "frozen split and annotations disagree"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"{split}.json"
    path.write_text(json.dumps(out))
    return path


def xywh_to_xyxy(b):
    x, y, w, h = b
    return np.array([x, y, x + w, y + h], np.float32)


def hflip_boxes(boxes, width):
    """Mirror [x1, y1, x2, y2] boxes left to right in an image `width` pixels wide."""
    boxes = np.asarray(boxes, np.float32).copy()
    x1 = boxes[:, 0].copy()
    boxes[:, 0] = width - boxes[:, 2]
    boxes[:, 2] = width - x1
    return boxes


class DetDataset(Dataset):
    """Returns (RGB image tensor in 0..1, target {boxes [x1,y1,x2,y2], labels = CarDD category id}, image_id).
    Train: horizontal flip with probability flip_p, boxes flipped too (D-026). Val/test: no augmentation."""

    def __init__(self, coco_json, img_dir, train, flip_p=0.5, seed=42):
        coco = json.loads(Path(coco_json).read_text())
        self.img_dir, self.train, self.flip_p, self.seed = Path(img_dir), train, flip_p, seed
        self.images = sorted(coco["images"], key=lambda im: im["id"])
        self.anns = {}
        for a in coco["annotations"]:
            self.anns.setdefault(a["image_id"], []).append(a)
        self.rng = None                                       # set per worker by worker_init_fn

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        im = self.images[i]
        bgr = cv2.imread(str(self.img_dir / im["file_name"]))
        if bgr is None:
            raise FileNotFoundError(self.img_dir / im["file_name"])
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        anns = self.anns.get(im["id"], [])
        boxes = np.array([xywh_to_xyxy(a["bbox"]) for a in anns], np.float32).reshape(-1, 4)
        labels = np.array([a["category_id"] for a in anns], np.int64)
        if self.train:
            if self.rng is None:
                self.rng = np.random.default_rng(self.seed)
            if self.rng.random() < self.flip_p:
                img = img[:, ::-1]
                boxes = hflip_boxes(boxes, img.shape[1])
        x = torch.from_numpy(np.ascontiguousarray(img).transpose(2, 0, 1).copy()).float().div_(255.0)
        return x, {"boxes": torch.from_numpy(boxes), "labels": torch.from_numpy(labels)}, int(im["id"])


def collate(batch):
    return tuple(zip(*batch))
