"""A6 classification data: frozen split (A4) + versioned crop cache (D-017) -> normalised tensors.
Train crops are augmented (D-021) then padded to 224; val crops are only padded. Test is locked (A12)."""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from vdv.data.augment import augment_train
from vdv.data.crops import check_crop_cache, crop_code_version, pad_to_square
from vdv.data.manifest import CLASSES, load_split

cv2.setNumThreads(0)                                   # no OpenCV thread pools inside DataLoader workers
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)   # RGB order
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
DEFAULT_CROP_DIR = Path("/content/data") / f"crops_{crop_code_version()}"


def to_tensor(img_bgr):
    """uint8 BGR image (H, W, 3) -> float32 RGB tensor (3, H, W), ImageNet-normalised."""
    rgb = img_bgr[:, :, ::-1].astype(np.float32) / 255.0          # OpenCV BGR -> RGB
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))


class CropDataset(Dataset):
    def __init__(self, rows, crop_dir, augment, seed=42, return_aug=False):
        self.rows, self.crop_dir, self.augment, self.seed = rows, Path(crop_dir), augment, seed
        self.return_aug = return_aug                                  # tests only: also return the brightness used
        self.labels = [CLASSES.index(r["cls"]) for r in rows]
        self.rng = None                                               # set per worker by worker_init_fn

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path = self.crop_dir / self.rows[i]["crop_path"]
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        params = {}
        if self.augment:
            if self.rng is None:                                      # main process (num_workers=0)
                self.rng = np.random.default_rng(self.seed)
            img, params = augment_train(img, self.rng)
        x = to_tensor(pad_to_square(img))
        if self.return_aug:
            return x, self.labels[i], i, params.get("brightness", 1.0)
        return x, self.labels[i], i


def worker_init_fn(worker_id):
    """Give each DataLoader worker its own augmentation generator. info.seed differs per worker and is
    derived from the DataLoader's seeded generator, so it is reproducible."""
    info = get_worker_info()
    ds = info.dataset
    while hasattr(ds, "dataset"):                                     # unwrap Subset(...)
        ds = ds.dataset
    ds.rng = np.random.default_rng(info.seed % 2**32)


def make_dataset(split, augment, seed=42, crop_dir=DEFAULT_CROP_DIR, allow_test=False):
    """The only way training code reads crops: frozen split + verified crop cache."""
    if split == "test" and not allow_test:
        raise RuntimeError("The test split is locked until the final evaluation (A12)")
    check_crop_cache(crop_dir)
    return CropDataset(load_split(split), crop_dir, augment, seed)
