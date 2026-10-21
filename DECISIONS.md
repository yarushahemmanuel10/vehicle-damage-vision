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
