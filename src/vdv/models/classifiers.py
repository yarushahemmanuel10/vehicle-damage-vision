"""A6 classifiers (D-022): ResNet-18 scratch / frozen / fine-tune, and ViT-B/16 linear probe."""
import torch.nn as nn
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights, resnet18, vit_b_16

KINDS = ["resnet18_scratch", "resnet18_frozen", "resnet18_finetune", "vit_b16_probe"]
FROZEN_KINDS = {"resnet18_frozen", "vit_b16_probe"}
HEAD_LR, BACKBONE_LR = 1e-3, 1e-4                     # D-022: head 1e-3, fine-tune backbone x0.1


def build_model(kind, num_classes=6, pretrained=True):
    """Returns (model, head, optimiser parameter groups). pretrained=False is for tests (no download)."""
    if kind.startswith("resnet18"):
        use_weights = pretrained and kind != "resnet18_scratch"
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if use_weights else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        head = model.fc
    elif kind == "vit_b16_probe":
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        head = model.heads.head
    else:
        raise ValueError(f"unknown model kind {kind!r}")

    head_ids = {id(p) for p in head.parameters()}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    if kind in FROZEN_KINDS:
        for p in backbone:
            p.requires_grad = False
        groups = [{"params": list(head.parameters()), "lr": HEAD_LR, "name": "head"}]
    elif kind == "resnet18_finetune":
        groups = [{"params": list(head.parameters()), "lr": HEAD_LR, "name": "head"},
                  {"params": backbone, "lr": BACKBONE_LR, "name": "backbone"}]
    else:                                                             # scratch: every layer at 1e-3
        groups = [{"params": list(model.parameters()), "lr": HEAD_LR, "name": "all"}]
    return model, head, groups


def set_train_mode(model, head, kind):
    """model.train(), except that a frozen backbone stays in eval mode, so its BatchNorm statistics
    and dropout do not change while only the head trains."""
    model.train()
    if kind in FROZEN_KINDS:
        model.eval()
        head.train()
