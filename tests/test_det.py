import json

import cv2
import numpy as np
from pytest import approx

from vdv.data.det_data import DetDataset, filter_coco, hflip_boxes, xywh_to_xyxy
from vdv.det_eval import coco_metrics
from vdv.models.detector import MAX_SIZE, MIN_SIZE, NUM_CLASSES, build_detector
from vdv.train_det import milestones

CATS = [{"id": i, "name": n} for i, n in
        enumerate(["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"], 1)]


def tiny_coco(tmp_path):
    img = np.zeros((50, 100, 3), np.uint8)
    img[10:20, 10:30] = 255                                   # white block = the box [10, 10, 20, 10]
    cv2.imwrite(str(tmp_path / "a.png"), img)
    coco = {"images": [{"id": 1, "file_name": "a.png", "width": 100, "height": 50}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 2, "bbox": [10, 10, 20, 10],
                             "area": 200, "iscrowd": 0}],
            "categories": CATS}
    p = tmp_path / "gt.json"
    p.write_text(json.dumps(coco))
    return p


def test_xywh_to_xyxy():
    assert xywh_to_xyxy([10, 20, 30, 40]).tolist() == [10, 20, 40, 60]


def test_hflip_boxes_and_twice_is_identity():
    b = np.array([[10, 20, 30, 40]], np.float32)
    f = hflip_boxes(b, 100)
    assert f.tolist() == [[70, 20, 90, 40]] and np.allclose(hflip_boxes(f, 100), b)


def test_flip_moves_pixels_and_boxes_together(tmp_path):
    ds = DetDataset(tiny_coco(tmp_path), tmp_path, train=True, flip_p=1.0)
    x, t, _ = ds[0]
    x1, y1, x2, y2 = t["boxes"][0].round().int().tolist()
    assert [x1, y1, x2, y2] == [70, 10, 90, 20]
    assert (x[:, y1:y2, x1:x2] > 0.9).all() and x[:, 30:, :].max() < 0.1


def test_no_flip_outside_training(tmp_path):
    _, t, img_id = DetDataset(tiny_coco(tmp_path), tmp_path, train=False)[0]
    assert t["boxes"][0].tolist() == [10, 10, 30, 20] and t["labels"].tolist() == [2] and img_id == 1


def test_filter_coco_drops_photos_and_their_annotations():
    coco = {"images": [{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "b.jpg"}],
            "annotations": [{"id": 1, "image_id": 1}, {"id": 2, "image_id": 2}, {"id": 3, "image_id": 2}],
            "categories": CATS}
    out = filter_coco(coco, ["a.jpg"])
    assert [im["id"] for im in out["images"]] == [1] and [a["id"] for a in out["annotations"]] == [1]


def test_milestones_match_the_paper_schedule():
    assert milestones(24) == [16, 22] and milestones(12) == [8, 11]


def test_detector_head_and_resolution():
    m = build_detector(pretrained=False)
    assert m.roi_heads.box_predictor.cls_score.out_features == NUM_CLASSES == 7
    assert m.transform.min_size == (MIN_SIZE,) and m.transform.max_size == MAX_SIZE


def test_coco_metrics_perfect_detection(tmp_path):
    m = coco_metrics(tiny_coco(tmp_path), [{"image_id": 1, "category_id": 2, "bbox": [10, 10, 20, 10], "score": 0.99}])
    # pycocotools averages over 101 recall points, so "perfect" is 1.0 only up to floating-point rounding
    assert m["mAP_50"] == approx(1.0) and m["mAP_50_95"] == approx(1.0) and m["per_class_AP"]["scratch"] == approx(1.0)
    assert m["per_class_AP"]["dent"] is None and m["AP_large"] is None     # no ground truth there
