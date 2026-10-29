# Vehicle Damage Vision

Classifying and locating car body damage in photos with PyTorch, built as a portfolio project for the ABL 1 Touch / University of Portsmouth KTP (ML/AI Computer Vision Engineer).

**Author:** Yarushah Emmanuel Sardar · **Dataset:** [CarDD](https://cardd-ustc.github.io/) (non-commercial research licence; no images or annotations in this repository)

---

## In plain English

A repair estimator looks at photos of a damaged car and decides what kind of damage each area shows. This project asks how much of that first step a model can take on safely, and where a person still needs to check.

- **Classifying damage.** Given a photo region around one damage, a fine-tuned ResNet-18 names the damage type (dent, scratch, crack, glass shatter, broken lamp or flat tyre) with a **test macro F1 of 0.937 ± 0.008** (mean ± SD over three seeds).
- **A review rule an estimator could use.** With a confidence threshold fixed in advance on validation data, **23% of damage regions go to a person for review**, and the remaining 77% are labelled automatically at **97.7% accuracy** on the unseen test set.
- **What the rule cannot catch.** The model's most confident mistakes (confidence above 0.9997) slip past any threshold. Some of them may be labelling gaps in the data rather than model errors. This matters more to a repair business than the average score, because these are the mistakes nobody would check.
- **Finding damage.** A Faster R-CNN detector trained to draw boxes around damage reaches a **validation mAP@50:95 of 0.493**. Its one-off test evaluation is still to be run (see [Status](#status)).

## Results

All numbers below come from committed files in `outputs/metrics/`. The test set was used once, after every choice had been fixed on validation.

### Phase A: damage-type classification (one crop per annotated damage)

Macro F1, mean ± SD over three seeds (seed 42 is the model carried forward):

| Method | Validation | Test |
|---|---|---|
| ResNet-18 trained from scratch | 0.797 ± 0.010 | 0.803 ± 0.011 |
| ResNet-18, frozen backbone (ImageNet) | 0.849 ± 0.003 | 0.871 ± 0.004 |
| **ResNet-18, fine-tuned (ImageNet)** | **0.930 ± 0.004** | **0.937 ± 0.008** |
| ViT-B/16 linear probe (frozen, ImageNet) | 0.870 ± 0.002 | 0.884 ± 0.002 |

- Test set: 785 crops from 374 photos. Seed-42 fine-tune: macro F1 0.928, 95% CI [0.908, 0.946] (photo-level bootstrap, 2,000 resamples). The paired bootstrap intervals for the differences between methods all exclude zero.
- Per-class test F1 (fine-tune, seed 42): dent 0.893 · scratch 0.920 · crack 0.842 · glass shatter 0.971 · lamp broken 0.958 · tire flat 0.984.
- Most test errors are between dent, scratch and crack: at least 56 of the 66, most often a dent labelled as a scratch (23).
- My predictions, written before training, got the ranking of the four methods exactly right but underestimated every absolute score by 7 to 20 points. The comparison and my reflection are in `docs/PREDICTIONS.md`.

### Human-in-the-loop review threshold

A crop goes to review when the model's top probability is below a cut-off. The rule, set before the test run: pick the lowest cut-off whose automatic labels reach at least 97% accuracy on validation. It chose **0.98**.

| | Sent to review | Accuracy of automatic labels |
|---|---|---|
| Validation | 25.2% | 97.07% |
| Test | 23.3% | 97.67% |

Calibration on validation: expected calibration error 0.048 (10 bins). The model is overconfident, which is why the review rule is set on observed accuracy, not on the raw confidence values.

### Mixed damage

Crops that contain at least half of another damage's box are harder: test accuracy 0.880 (n = 75) against 0.920 for clean crops (n = 710). The same pattern holds on validation (0.829 against 0.921).

### Phase B: damage detection (validation only so far)

Faster R-CNN ResNet-50-FPN (torchvision, COCO-pretrained), 12 epochs, best epoch 12, 805 validation photos:

| mAP@50:95 | mAP@50 | mAP@75 | AP small | AP medium | AP large |
|---|---|---|---|---|---|
| 0.493 | 0.682 | 0.533 | 0.099 | 0.105 | 0.495 |

Per-class AP (validation): dent 0.239 · scratch 0.256 · crack 0.170 · glass shatter 0.879 · lamp broken 0.619 · tire flat 0.796.

For context only: the CarDD paper reports a box mAP@50:95 of 49.2 for Mask R-CNN R50 on the **test** set, trained for 24 epochs jointly with masks (Wang et al., Tables IV–V). These validation numbers are not a like-for-like comparison.

<!-- When B5 has run, replace this section's heading with "Phase B: damage detection (test)" and add the test table from outputs/metrics/det_test_results.json, side by side with the paper's figures and caveats. -->

## Status

- **Phase A (classification):** complete, including the one-off test evaluation and error analysis.
- **Phase B (detection):** trained and validated. The one-off test evaluation script (`scripts/eval_det_test.py`) is committed and guarded: it re-checks validation before touching the test set, and refuses to run twice. On its first attempt, the validation check came out 0.0021 away from the recorded value (tolerance 0.002), so it stopped before the test set. The cause is being investigated. **The test set has not been used for detection yet.**

## How it was evaluated

- **The test set is used once** per phase, after every setting and threshold has been fixed on validation.
- **Three seeds per method,** reported as mean ± SD, with photo-level bootstrap confidence intervals (crops from the same photo are resampled together, because they are not independent).
- **Predictions written before each run** (`docs/PREDICTIONS.md`), with actuals appended afterwards and the predictions never edited. Git timestamps show the order.
- **Data checks before training:** annotation verification, a perceptual-hash duplicate check across splits (including mirrored copies), and a frozen, hashed crop manifest. These found annotation IDs repeated across splits (a silent leakage risk) and a cropped duplicate between validation and training.
- **Every decision logged** in `DECISIONS.md` (each stating who decided), and **every mistake caught logged** in `CORRECTIONS.md`.

## Dataset and licence

This project uses **CarDD** (Wang, Li and Wu, *CarDD: A New Dataset for Vision-Based Car Damage Detection*, IEEE Transactions on Intelligent Transportation Systems, 2023, [arXiv 2211.00945](https://arxiv.org/abs/2211.00945)): 4,000 photos, 8,740 annotated damages (counted from the annotation files; the paper describes "over 9,000"), six classes, official split 2,816 / 810 / 374.

CarDD is released for non-commercial research and educational use, and may not be redistributed. **No images or annotations are included here,** and neither are files derived from them (the crop manifest and per-crop prediction files); these regenerate from the scripts once you have the dataset. To reproduce the results, request CarDD from the authors and set it up as described in `notebooks/00_start_session.ipynb`. The full private repository is available on request.

## How to reproduce

Developed on Google Colab (Tesla T4), Python 3.13, PyTorch 2.11.0, torchvision 0.26.0.

1. Get CarDD from the authors (see above).
2. Open `notebooks/00_start_session.ipynb` in Colab and run it. It installs the package (`pip install -e .`), checks the PyTorch build and unpacks the data.
3. Run the pipeline in order:
   - **Data:** `scripts/verify_data.py`, `box_stats.py`, `build_crops.py`, `dup_calibrate.py`, `dup_check.py`
   - **Classification:** `python -m vdv.train_cls` (four methods × three seeds), then `scripts/review_threshold.py` and `scripts/eval_test.py`
   - **Detection:** `python -m vdv.train_det`, then `scripts/eval_det_test.py`
4. `pytest` runs the test suite.

## Limitations

- **Crops, not whole photos.** The classifier sees a region centred on one known damage. Finding the damage is the detector's job, and the two are not yet joined into one pipeline.
- **No "undamaged" class.** The classifier assumes every crop contains damage, which a real workflow cannot assume.
- **Overconfidence.** The confidence scores are not calibrated (ECE 0.048), and the most confident errors pass any threshold. The review threshold was also tuned on validation, so its test behaviour is an estimate.
- **Possible label gaps.** In 2 of the 4 most confident test errors, the crop visibly appears to contain dent damage that is not annotated. This is based on visual inspection of a small sample, not an established fact.
- **Single-label crops.** About 12% of training crops contain at least half of another damage's box, so one label per crop does not always describe what the crop shows.
- **One public dataset.** CarDD photos are web images, 98% of them close-ups of part of a car. They are not ABL's photos, so the results will not transfer unchanged to ABL's sites, cameras and repair records.
- **Small test set.** 785 crops from 374 photos; the confidence intervals show the uncertainty (flat tyres, for example, have only 32 test crops).
- **Duplicate check limits.** Perceptual hashing misses copies cropped beyond light trimming, and watermarked copies; only the closest pairs above the threshold were checked by eye, and one of those ten was a cropped duplicate. Different photos of the same vehicle cannot be detected this way.
- **Detection trained for 12 epochs,** half the paper's schedule; validation mAP was still rising slightly at the end.
- **Small-object detection is weak** (AP small 0.099 on validation), which is where cracks and fine scratches sit. AP medium (0.105) is also far below the paper's figure and is not yet explained.

## How this would extend at a repair business

These are future work, not built here.

- **Start from ABL's own annotated repair records,** with labelling guidance agreed with estimators for overlapping damage types.
- **Join detection and classification** into one photo-to-estimate pipeline, with an "undamaged" class.
- **Calibrate before thresholding** (for example temperature scaling), so a review dial means what it says.
- **Learn from the review queue:** every estimator correction becomes new labelled data, which is the human-in-the-loop learning the KTP describes.
- **Monitor by site and camera,** since performance can drift between branches and devices.

## How this project was built

I directed this project and made its design decisions. I wrote the brief and the working rules: one approved step at a time, the test set locked until a single final run, predictions written before every training run, and every reported number traceable to a script in this repository.

An AI coding assistant proposed options at each step and wrote code to my specification. I chose between the options, reviewed every change before it was committed, and ran every training and evaluation step myself on Google Colab. I was new to PyTorch, OpenCV, transfer learning and object detection when I started, so I kept plain-English notes on every new concept in `docs/LEARNING_NOTES.md`.

The logs are the evidence:

- `DECISIONS.md` records each design choice, the alternatives, the reason, and who decided: me, or an option proposed to me that I accepted or changed.
- `CORRECTIONS.md` records problems I caught before they affected results, from a `.gitignore` rule that would have hidden the code folder to training runs that could not have been restarted after a disconnect.
- `docs/PREDICTIONS.md` holds my predictions, committed before each training run, the actual results added afterwards, and my reflection on the difference.

## Repository layout

```
src/vdv/            package: data checks, crops, duplicates, datasets, models, training, evaluation
scripts/            one script per pipeline step
tests/              pytest suite
outputs/metrics/    every reported number, as JSON
outputs/figures/    charts (no dataset images)
notebooks/          start-of-session notebook for Colab
docs/               data report, predictions and reflection, learning notes, Phase C checklist
DECISIONS.md        design decisions and who made them
CORRECTIONS.md      mistakes found and how they were fixed
```
