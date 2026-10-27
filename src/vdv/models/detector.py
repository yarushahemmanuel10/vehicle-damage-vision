"""Phase B detector (D-026): torchvision Faster R-CNN ResNet-50-FPN v1, COCO-pretrained, boxes only."""
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

NUM_CLASSES = 7                 # background (0) + 6 damage classes; labels are CarDD category ids 1..6
MIN_SIZE, MAX_SIZE = 800, 1000  # D-026: most 1000-px-wide photos stay at native size


def build_detector(pretrained=True):
    """pretrained=False is for tests (no download)."""
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None,
                                    weights_backbone=None, min_size=MIN_SIZE, max_size=MAX_SIZE)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model
