# Decisions log

One entry per design choice. "Decided by" is one of: *me* · *AI proposed / I accepted* · *AI proposed / I changed*.

---

### D-001 · Dataset: CarDD for both classification and detection
- **Date:** 2026-10-21
- **Choice:** CarDD (Wang et al., IEEE T-ITS 2023): 4,000 images, 6 damage classes, boxes and masks.
- **Alternatives:** Kaggle binary damaged/whole set (no damage types, licence unclear); VehiDE (too large for the timeline, licence unclear); Roboflow Universe CC BY 4.0 sets (variable label quality, duplicate risk).
- **Reason:** One dataset for both phases saves a day and links them: classify damage, then locate it, on the same images. Licence obtained via signed form; access granted 2026-10-21. Use is non-commercial research and learning.
- **Decided by:** AI proposed / I accepted

### D-002 · Classification framing: crop-level, single label
- **Date:** 2026-10-21
- **Choice:** Classify crops of each annotated damage region into one of 6 types.
- **Alternatives:** single-class images only; dominant class per image; multi-label per image.
- **Reason:** Matches how an estimator assesses damage (one damage at a time), uses all annotated instances, gives clean single-label metrics, and joins with detection into one pipeline (detect → classify → review). Known weakness: training crops come from perfect human boxes; neighbouring damage can appear in a crop.
- **Decided by:** AI proposed / I accepted

### D-003 · Compute and storage: Colab T4, code on GitHub, data on Google Drive
- **Date:** 2026-10-21
- **Choice:** Train on Colab (Tesla T4, 15 GB). Clone the repo to Colab's local disk each session; keep dataset and checkpoints on Google Drive.
- **Alternatives:** Kaggle Notebooks (kept as backup); local CPU only (too slow for fine-tuning, ViT and detection).
- **Reason:** Colab disk is wiped each session, Drive persists; git is unreliable on a mounted Drive.
- **Decided by:** AI proposed / I accepted

### D-004 · Detection library: torchvision, not MMDetection
- **Date:** 2026-10-21
- **Choice:** torchvision Faster R-CNN (Mask R-CNN only as a stretch goal).
- **Alternatives:** MMDetection (likely used for the CarDD paper baselines); Ultralytics YOLO (AGPL-3.0).
- **Reason:** MMDetection 2.x needs mmcv compiled against old PyTorch/CUDA, a high install risk on torch 2.11 / CUDA 13. Its config-driven training hides the loop I need to explain. torchvision has no extra installs and a permissive licence.
- **Decided by:** AI proposed / I accepted

### D-005 · No dataset images in git
- **Date:** 2026-10-21
- **Choice:** `/data/`, all `*.jpg`/`*.jpeg` and `/outputs/figures_private/` are git-ignored. Only charts without photos are committed.
- **Alternatives:** commit small thumbnails.
- **Reason:** The CarDD licence forbids distributing any part of the dataset to third parties.
- **Decided by:** AI proposed / I accepted

### D-006 · Split: CarDD official split, with near-duplicate removal
- **Date:** 2026-10-22
- **Choice:** Use the authors' official image-level train/val/test split (sizes to be confirmed by the A1 verification script). Run a perceptual-hash near-duplicate check on test↔train and val↔train. The Hamming-distance threshold is set by me before any results are seen; flagged pairs are reviewed by eye. Duplicates are removed from val/test only; the data is never re-split. The number removed is recorded as a deviation from the official split.
- **Alternatives:** our own stratified or vehicle-grouped split; the official split without a duplicate check.
- **Reason:** Results are comparable with the paper's published baselines, and the test set was fixed by the authors before I saw the data. The split is by image, so crops from one photo cannot cross splits. It is by file, not by vehicle, hence the duplicate check.
- **Decided by:** AI proposed / I changed: I limited the check to test↔train and val↔train, required the threshold to be fixed before seeing results, and added manual review of flagged pairs.

### D-007 · Workflow: Colab only, code written and committed from Colab
- **Date:** 2026-10-22
- **Choice:** All code lives in the `src/vdv/` package, written block by block from Colab with `%%writefile` and committed and pushed from Colab. Tests run in Colab on a tiny sample, on CPU where practical. Kaggle Notebooks are the backup.
- **Alternatives:** edit on my Mac, push to GitHub and pull in Colab; local CPU tests (the original plan).
- **Reason:** I run and review every block myself in one environment. This avoids version mismatches between my Intel Mac (limited to older PyTorch builds) and Colab. It replaces the original local-CPU-test plan.
- **Decided by:** me

- **Amendment (2026-10-23):** Any later `pip install` must use `-c /content/constraints.txt`, so no package can change PyTorch. If the start-of-session torch check fails, the fallback is to stay on Colab and install the tested build (`torch==2.11.0+cu130`, `torchvision==0.26.0+cu130`) from PyTorch's own cu130 package index. If that build cannot run on Colab's GPU driver, stop and decide again before any training.
- **Amendment decided by:** me

### D-008 · Licence: using CarDD for this project counts as educational use
- **Date:** 2026-10-22
- **Choice:** I use CarDD only to educate myself: to learn PyTorch, transfer learning, object detection and vision transformers, the skills the KTP role requires. The code, metrics and write-up may be shown in my job application. No dataset images are redistributed (D-005), no commercial system is tested, and CarDD would not be used in any actual KTP or company work without the PIC Lab's authorisation.
- **Alternatives:** treat portfolio use as commercial and ask the PIC Lab for explicit permission; switch to a CC BY 4.0 dataset.
- **Reason:** The project's purpose is my own learning for the KTP. That is educational use, which the licence permits and encourages. Showing my own code and results is not distributing the dataset.
- **Decided by:** me

### D-009 · Evaluation libraries: pycocotools and scikit-learn, no torchmetrics
- **Date:** 2026-10-22
- **Choice:** pycocotools (the official COCO evaluator) for detection: mAP@50, mAP@50:95, AP by object size and per-class AP. scikit-learn for classification metrics; numpy for bootstrap confidence intervals.
- **Alternatives:** torchmetrics (a PyTorch wrapper offering COCO-style mAP and classification metrics).
- **Reason:** Most likely the same evaluator used for the CarDD paper baselines, which keeps that comparison fair. Already installed, covers AP by object size, and avoids one dependency. Trade-off: I write the converter from model predictions to COCO JSON myself, so I know exactly what is scored.
- **Decided by:** AI proposed / I accepted

### D-010 · Seeds and run-to-run variation
- **Date:** 2026-10-23
- **Choice:**
  - Three seeds per ResNet-18 run (scratch, frozen, fine-tune).
  - cuDNN deterministic mode off; seed variation is measured instead.
  - ViT run: 1 seed, or 3 if the timing allows.
  - The A6 smoke test times one full training epoch before the run budget is fixed.
  - Report mean ± standard deviation across seeds, on validation and test.
  - Seed 42 is the model carried forward to the review-threshold table, test examples and Grad-CAM. The other seeds only measure variation.
  - If time is short: cut from 3 seeds to 2, then to 1 with the limitation stated.
- **Alternatives:** one seed per run; deterministic mode on.
- **Reason:** Seed variation shows whether the gaps between methods are real. Bootstrap confidence intervals separately measure test-set sampling uncertainty.
- **Decided by:** me
