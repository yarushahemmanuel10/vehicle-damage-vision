# Predictions (written before any training run)

Scratch ResNet-18: val macro F1 0.45–0.60. No pre-trained features, and only ~6,200 crops and 20 epochs; still improving at epoch 20.

Frozen ResNet-18: 0.62–0.72. ImageNet features capture general shape and texture, but not paint damage specifically; only the head adapts.

Fine-tune ResNet-18: 0.75–0.85. Adapting the backbone to damage textures should add 5–10 points over frozen.

ViT-B/16 linear probe: 0.70–0.80. A larger model with richer frozen features; should beat the frozen ResNet, but probably not the fine-tuned one.

Ranking: fine-tune > ViT probe > frozen > scratch. Least certain: ViT probe vs fine-tune.

Hardest class: crack (tiny, thin, 44% needed the minimum window, often inside other damage). Main confusion: dent ↔ scratch (ambiguous labels seen on the contact sheet). Easiest: glass shatter and tire flat (large and distinctive).

Seed spread: about ±1 point for frozen and probe runs (only the head and data order are random); a few points for scratch. The gaps between methods should exceed the spread, except possibly ViT probe vs frozen ResNet.

Note: informed by general transfer-learning expectations discussed beforehand.

## Actual results (validation; added after the runs, predictions above unchanged)

Source: `outputs/metrics/val_summary.json` and `outputs/metrics/runs/*/val_probs.csv`, best epoch per run (D-022). Validation also chose each run's best epoch, so these figures are slightly optimistic; the test evaluation (A12) is the unbiased one.

| Model | Predicted val macro F1 | Actual, mean ± SD over 3 seeds | Versus prediction |
|---|---|---|---|
| Scratch ResNet-18 | 0.45–0.60 | 0.797 ± 0.010 | above the range |
| Frozen ResNet-18 | 0.62–0.72 | 0.849 ± 0.003 | above the range |
| Fine-tune ResNet-18 | 0.75–0.85 | 0.930 ± 0.004 | above the range |
| ViT-B/16 linear probe | 0.70–0.80 | 0.870 ± 0.002 | above the range |

- **Ranking:** predicted Fine-tune ResNet-18 > ViT-B/16 linear probe > Frozen ResNet-18 > Scratch ResNet-18; actual Fine-tune ResNet-18 > ViT-B/16 linear probe > Frozen ResNet-18 > Scratch ResNet-18 (same order).
- **Fine-tune minus frozen:** predicted 5 to 10 points; actual 8.1 points.
- **ViT probe minus frozen ResNet:** 2.1 points; seed SDs 0.2 and 0.3 points.
- **Seed spread (SD, points):** Scratch ResNet-18 1.0, Frozen ResNet-18 0.3, Fine-tune ResNet-18 0.4, ViT-B/16 linear probe 0.2.

**Per-class validation F1 (mean over 3 seeds):**

| Model | dent | scratch | crack | glass shatter | lamp broken | tire flat | Lowest |
|---|---|---|---|---|---|---|---|
| Scratch ResNet-18 | 0.710 | 0.743 | 0.632 | 0.871 | 0.865 | 0.961 | crack |
| Frozen ResNet-18 | 0.771 | 0.805 | 0.739 | 0.936 | 0.865 | 0.976 | crack |
| Fine-tune ResNet-18 | 0.871 | 0.900 | 0.879 | 0.968 | 0.970 | 0.992 | dent |
| ViT-B/16 linear probe | 0.801 | 0.830 | 0.769 | 0.981 | 0.871 | 0.968 | crack |

**Most frequent confusions (true → predicted, summed over 3 seeds):**

- Scratch ResNet-18: scratch → dent (397); dent → scratch (268); scratch → crack (197)
- Frozen ResNet-18: scratch → dent (262); dent → scratch (241); scratch → crack (166)
- Fine-tune ResNet-18: dent → scratch (201); scratch → dent (121); crack → scratch (47)
- ViT-B/16 linear probe: scratch → dent (229); dent → scratch (213); scratch → crack (143)

## Actual results (test; added after the one-off test evaluation, predictions above unchanged)

Source: `outputs/metrics/test_results.json` (commit `db685d1`). Test: 785 crops from 374 photos. My predictions were made for validation macro F1; they are repeated here for reference.

| Model | Predicted (val) | Test macro F1, mean ± SD over 3 seeds | Seed 42 test macro F1 [95% CI, photo bootstrap] |
|---|---|---|---|
| Scratch ResNet-18 | 0.45–0.60 | 0.803 ± 0.011 | 0.801 [0.772, 0.829] |
| Frozen ResNet-18 | 0.62–0.72 | 0.871 ± 0.004 | 0.866 [0.841, 0.890] |
| Fine-tune ResNet-18 | 0.75–0.85 | 0.937 ± 0.008 | 0.928 [0.908, 0.946] |
| ViT-B/16 linear probe | 0.70–0.80 | 0.884 ± 0.002 | 0.885 [0.862, 0.906] |

**Paired differences, seed 42 (same 2000 photo resamples), 95% CI:**

- Fine-tune ResNet-18 minus Scratch ResNet-18: macro F1 [+0.098, +0.157], accuracy [+0.122, +0.184]
- Fine-tune ResNet-18 minus Frozen ResNet-18: macro F1 [+0.039, +0.084], accuracy [+0.050, +0.100]
- Fine-tune ResNet-18 minus ViT-B/16 linear probe: macro F1 [+0.018, +0.069], accuracy [+0.031, +0.082]

**Per-class test F1, fine-tune seed 42:**

| dent | scratch | crack | glass shatter | lamp broken | tire flat |
|---|---|---|---|---|---|
| 0.893 (n=236) | 0.920 (n=307) | 0.842 (n=70) | 0.971 (n=71) | 0.958 (n=69) | 0.984 (n=32) |

**Review at the operating threshold 0.98 (fine-tune seed 42):** 183 of 785 crops to review (23.3%); auto-labelled accuracy 0.977, macro F1 0.981.

**D-019 clean vs contaminated (fine-tune seed 42):**

- val, clean (other_class_ge50=0): accuracy 0.921, macro F1 0.942 (n = 1508 crops, 800 photos)
- val, contaminated (other_class_ge50=1): accuracy 0.829, macro F1 0.835 (n = 228 crops, 167 photos)
- test, clean (other_class_ge50=0): accuracy 0.920, macro F1 0.929 (n = 710 crops, 374 photos)
- test, contaminated (other_class_ge50=1): accuracy 0.880, macro F1 0.864 (n = 75 crops, 57 photos)

## My reflection

_(to be written by me)_
