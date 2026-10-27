"""Phase B COCO evaluation with pycocotools (D-009): mAP@50:95, mAP@50, AP by size, per-class AP."""
import contextlib
import io

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

KEYS = ["mAP_50_95", "mAP_50", "mAP_75", "AP_small", "AP_medium", "AP_large"]


def to_coco_detections(outputs, image_ids):
    """torchvision outputs ([x1,y1,x2,y2] boxes) -> COCO result dicts ([x, y, w, h])."""
    dets = []
    for out, img_id in zip(outputs, image_ids):
        boxes = out["boxes"].float().cpu().numpy()
        scores = out["scores"].float().cpu().numpy()
        labels = out["labels"].cpu().numpy()
        for (x1, y1, x2, y2), s, l in zip(boxes, scores, labels):
            dets.append({"image_id": int(img_id), "category_id": int(l),
                         "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)})
    return dets


def coco_metrics(gt_path, detections):
    """Box metrics. AP by size uses the ground-truth 'area' field (mask area), as pycocotools does.
    Per-class AP: IoU 0.50:0.95, all areas, 100 detections. None where a class or size has no ground truth."""
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_path))
        names = {c["id"]: c["name"] for c in gt.loadCats(gt.getCatIds())}
        if not detections:
            return {**{k: 0.0 for k in KEYS}, "per_class_AP": {n: 0.0 for n in names.values()}, "n_detections": 0}
        ev = COCOeval(gt, gt.loadRes(detections), "bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
    stats = {k: (None if ev.stats[i] < 0 else float(ev.stats[i])) for i, k in enumerate(KEYS)}
    per_class = {}
    for k, cat_id in enumerate(ev.params.catIds):
        p = ev.eval["precision"][:, :, k, 0, -1]
        p = p[p > -1]
        per_class[names[cat_id]] = float(p.mean()) if p.size else None
    return {**stats, "per_class_AP": per_class, "n_detections": len(detections)}
