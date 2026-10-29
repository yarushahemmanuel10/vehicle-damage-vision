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


## Phase B predictions (written before training)

Test box mAP@50:95: 0.42–0.50. mAP@50: 0.60–0.68. Expected slightly below the paper's Mask R-CNN R50 box AP (49.2 / 66.3): Faster R-CNN has no mask supervision to sharpen its boxes, and the run is 12 epochs instead of 24.

Per-class AP: highest for glass shatter and tire flat (above 0.75: large, distinctive, clear boundaries); lamp broken in the middle (~0.5); lowest for crack (below 0.25: thin, small, ambiguous extent), with dent and scratch also low (0.2–0.35), because their boundaries are fuzzy and IoU punishes loose boxes.

AP by size: small very low (0.10–0.20), large highest (0.50–0.60), following the paper's pattern.

Classification vs detection: detection scores will be far lower than the 0.94 classification macro F1, because detection must also find and tightly localise damage whose extent is ambiguous.

Training curve: validation mAP improves quickly in the first epochs, with a visible jump after the first learning-rate drop (epoch 8), then plateaus.

Note: written before training; informed by the paper's published baselines and general expectations discussed beforehand.

## My reflection

Phase A: the ranking was right, the levels were not.
I predicted the order exactly: fine-tune > ViT probe > frozen > scratch. The fine-tune’s gain over the frozen backbone (8.1 points on validation) landed inside my 5–10 range, and scratch was still improving at epoch 20, as I expected. But every model scored above the top of my range, by 7 to 20 points. Scratch alone reached 0.80 validation macro F1, against my 0.45–0.60.

I think I misjudged the difficulty of the task the model actually saw. I was imagining whole-photo damage classification, closer to harder benchmarks. In fact, each input was a crop centred on one annotated damage, with only a small margin of context. Several classes are visually distinctive at that scale (glass shatter, flat tyres, broken lamps), the crop pipeline made even tiny cracks readable through the minimum window, and ImageNet pre-training gave more than I credited. Getting the ranking right while missing the levels suggests I understood the relative effects of each method better than I could predict absolute performance. That is a useful thing to know about my own judgement.

The seed spread was also smaller than I expected (0.2–0.3 points for the frozen models, 1 point for scratch), and the ViT probe beat the frozen ResNet by a clear margin, where I had thought the difference might be lost in the noise.

What the test confirmed.

The test results matched validation: the fine-tune scored 0.937 on test against 0.930 on validation, so choosing epochs and the threshold on validation showed no sign of inflating the results.
Crack was the hardest class (F1 0.84, recall 0.80). Glass shatter (0.97) and flat tyres (0.98) were the easiest. Most errors ran between crack, dent and scratch, as I expected: at least 56 of the 66 test errors for the fine-tuned model, most often dent mistaken for scratch (23).
At the threshold my rule picked in advance (0.98), 23% of test crops went to review, and the auto-labelled share reached 97.7% accuracy, meeting my 97% target on unseen data.
I limited the brightness augmentation out of concern for light-coloured cars. On test, they scored slightly better than dark and coloured cars (the small unknown-colour group, 34 crops, scored higher still; no confidence intervals for subgroups), so that worry was not borne out.

What surprised me.

Confident errors. The model’s four most confident test errors all had confidence above 0.9997, so no review threshold could have caught them. Validation had already shown overconfidence (ECE 0.048, and validation loss rising while F1 stayed flat), but seeing concrete cases made the limitation clear: a review queue catches uncertain predictions, not confidently wrong ones.
Mixed damage. Crops containing at least half of another damage type’s box were harder on both validation and test (test accuracy 0.88 against 0.92 for clean crops). But none of the four confident errors was flagged that way. Looking at them, two visibly appear to contain dent damage that isn’t annotated as a separate box. Some errors may come from labels as much as from the model, though that’s based on a small sample.

What a repair business would care about.
The headline for ABL is the trade-off: at 23% review, the remaining 77% of crops were labelled at 97.7% accuracy. That’s a real reduction in estimator workload, with a clear dial for how much risk to accept. The confident errors matter more than the average score, though, because they are exactly the mistakes nobody would check. The model also has no “undamaged” class: it assumes every crop contains damage, which a real workflow cannot assume.

What I would do differently.

Calibrate before setting a review threshold, for example with temperature scaling, so confidence means what it says.
Handle mixed damage directly: multi-label classification or detection, plus clearer labelling guidance for damage types that overlap.
Estimate difficulty from the model’s point of view (the actual crop, not the original photo), and anchor predictions against a known baseline before writing them.
Plan time for data preparation realistically. Verification, crops and the duplicate check were worth doing, and they caught real problems (repeated annotation IDs across splits, a cropped duplicate), but they took longer than planned and forced me to cut scope.
Use a stronger duplicate check: feature-based matching alongside the perceptual hash, since the hash cannot see cropped copies and my spot-check found one.

Phase B: (to add after the detection results)
