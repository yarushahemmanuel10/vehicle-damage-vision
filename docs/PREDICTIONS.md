# Predictions (written before any training run)

Scratch ResNet-18: val macro F1 0.45–0.60. No pre-trained features, and only ~6,200 crops and 20 epochs; still improving at epoch 20.

Frozen ResNet-18: 0.62–0.72. ImageNet features capture general shape and texture, but not paint damage specifically; only the head adapts.

Fine-tune ResNet-18: 0.75–0.85. Adapting the backbone to damage textures should add 5–10 points over frozen.

ViT-B/16 linear probe: 0.70–0.80. A larger model with richer frozen features; should beat the frozen ResNet, but probably not the fine-tuned one.

Ranking: fine-tune > ViT probe > frozen > scratch. Least certain: ViT probe vs fine-tune.

Hardest class: crack (tiny, thin, 44% needed the minimum window, often inside other damage). Main confusion: dent ↔ scratch (ambiguous labels seen on the contact sheet). Easiest: glass shatter and tire flat (large and distinctive).

Seed spread: about ±1 point for frozen and probe runs (only the head and data order are random); a few points for scratch. The gaps between methods should exceed the spread, except possibly ViT probe vs frozen ResNet.

Note: informed by general transfer-learning expectations discussed beforehand.
