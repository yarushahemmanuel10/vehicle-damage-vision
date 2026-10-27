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

- **Amendment (2026-10-27):** pHash distance ≤ 6 → duplicate, removed automatically. The threshold was set from the train calibration (`outputs/metrics/dup_calibration_train.json`): all JPEG-recompressed, resized and brightness-changed copies were at distance ≤ 4; the nearest different train photo was at 8; pHash distances are always even. No full review band. Comparisons: val↔train and test↔train (val/test photos also compared mirrored) and val↔test. Spot-check by eye: the 10 closest pairs above the threshold plus train pair 003010/001319; nothing from the spot-check is removed unless marked "dup"; decisions in `splits/dup_review.csv`. Removal: only the val/test photo, never train; for val↔test duplicates, the val photo; removal flags crops in the manifest (`dup_excluded`) and never touches files on disk.
- **Amendment limitation:** pHash misses copies cropped beyond light trimming, and watermarks or overlays; the spot-check result is reported (`outputs/metrics/dup_summary.json`).
- **Amendment decided by:** me

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

### D-011 · CarDD_SOD folder: present but out of scope
- **Date:** 2026-10-23
- **Choice:** Use only `CarDD_COCO`. The `CarDD_SOD` folder (salient object detection: copies of the same images with binary masks and edge maps) is not used.
- **Alternatives:** use the SOD masks for segmentation.
- **Reason:** Both tasks need a class label per damage, which only the COCO annotations provide. COCO also contains segmentation masks for every annotation, for the Mask R-CNN stretch goal. SOD images are copies, so reading them would double-count images.
- **Decided by:** AI proposed / I accepted

### D-012 · image_info.xlsx: subgroup analysis and cross-check only
- **Date:** 2026-10-23
- **Choice:**
  - Use `image_info.xlsx` for subgroup error analysis and as a consistency cross-check only, not as a model input.
  - Bins fixed now, before any results: colour = light (white, silver, champagne) / dark (black, gray) / coloured (all other named colours, including "mixture") / unknown (blank); angle = front / side / rear / other-unknown ("inside" and blank); completeness: counts only.
  - Report group sizes beside every subgroup figure.
  - Pin `openpyxl==3.1.5`; strip the trailing space in the `complete or partial ` column name.
  - Spreadsheet summary and the width/height/#instances/#categories cross-check live in `scripts/verify_data.py`, so the counts are citable.
- **Alternatives:** leave the spreadsheet out of scope; use it as a model input.
- **Reason:** Subgroup results (for example, by car colour or shooting angle) show where the model fails in ways a repair business would recognise. Fixing bins before results prevents choosing groups that make results look better.
- **Decided by:** me

### D-013 · Design statistics come from the train split only
- **Date:** 2026-10-23
- **Choice:** Statistics used to make design choices (crop margin, minimum crop size, resize method, augmentation settings) are computed on the train split only. The test split stays untouched until the final evaluation (A12); validation is used only for model selection and the review threshold.
- **Alternatives:** compute design statistics on all splits.
- **Reason:** Choices tuned on test data, even indirectly through its statistics, make the final test result optimistic.
- **Decided by:** AI proposed / I accepted

### D-014 · Crop margin: 10% of box size on each side
- **Date:** 2026-10-23
- **Choice:** Add 10% of the box width (left and right) and 10% of the box height (top and bottom), clipped to the image. A2 must report how many final crops contain a box of another class, measured on the final crop window (margin + 64 px minimum window + aspect cap), not the margin alone.
- **Alternatives:** no margin; 25%; a fixed number of pixels.
- **Reason:** On the train contact sheet, 10% adds a little edge context, while 25% pulls in noticeably more surrounding car body. 38% of train boxes already overlap a box of another class (`box_stats_train.json`), so context is kept small.
- **Decided by:** me

### D-015 · Minimum crop window: 64 px per side, all splits, nothing dropped
- **Date:** 2026-10-23
- **Choice:** After the margin, grow any side shorter than 64 px to exactly 64 px around the box centre, using real image context (shifted to stay inside the image). Applied to train, val and test; no boxes are dropped. A test checks the result is exactly 64 px.
- **Alternatives:** drop boxes with a shorter side below a threshold; apply the minimum to train only.
- **Reason:** Dropping boxes with a shorter side below 32 px would remove 27% of train cracks (`box_stats_train.json`). On the contact sheet, the 64 px window made the smallest cracks recognisable. Applying it to all splits keeps val/test results representative of tiny damage instead of removing the hardest cases.
- **Decided by:** me; minimum-window option AI-proposed

### D-016 · Resize: aspect-capped expansion, then pad to 224 × 224
- **Date:** 2026-10-23
- **Choice:** After margin and minimum window, grow the shorter side with real image context until long side / short side ≤ 3 (shifted to stay inside the image), then pad to 224 × 224 with mid-grey. Crops with aspect ≤ 3 are unchanged by this step. Verify on A2 sample crops: #5727, #1984 and the typical boxes.
- **Alternatives:** stretch; pad only; full square-expand; pad normally and square-expand above aspect 4.6.
- **Reason:** On the train contact sheet, stretch destroyed #5727 and #1984; pad left them as hairlines; full square-expand pulled a tail light into crack #1984. Aspect-capped expansion keeps real context but bounds it, leaves typical boxes unchanged, and avoids the jump at a single threshold and the context risk of a full square.
- **Decided by:** me; aspect-capped option AI-proposed

### D-017 · Crop storage, caching and manifest key
- **Date:** 2026-10-23
- **Choice:**
  - Crops are built once per crop-code version as PNG (aspect ratio kept, long side 224 px, unpadded) and zipped without compression to Drive as `crops_<version>.zip`; the start-of-session cell unzips the matching version.
  - The version is the hand-set `CROP_VERSION` plus a hash of `src/vdv/data/crops.py` and `scripts/build_crops.py`; `check_crop_cache()` refuses a cache built by different code.
  - Augmentation and padding to 224 × 224 happen at load time.
  - The crop manifest (`splits/crops_manifest.csv`) is keyed on (split, ann_id): image IDs are unique across the three annotation files, but annotation IDs are not (1744 are shared).
- **Alternatives:** make crops on the fly every epoch; store padded 224 × 224 crops; key on ann_id alone.
- **Reason:** Training reads small files instead of decoding full photos every epoch across 9 to 12 runs. The code hash makes it impossible to train silently on crops built by older rules. Storing unpadded crops lets augmentation such as rotation happen before padding.
- **Decided by:** me

### D-018 · Validation and test crop statistics are reported only
- **Date:** 2026-10-23
- **Choice:** `crops_summary.json` reports crop counts and flags for every split, including test. The D-014 to D-016 crop rules are fixed and are not adjusted on test-split evidence (D-013).
- **Alternatives:** report train statistics only.
- **Reason:** Reporting all splits documents the data; adjusting rules on test evidence would make the final test result optimistic.
- **Decided by:** me

### D-019 · Subgroup analysis: clean vs contaminated crops
- **Date:** 2026-10-23
- **Choice:** Report the classification metrics on val and test separately for crops with `other_class_ge50` = 1 (at least half of an other-class box inside the final crop window) and = 0. Reported overall only, not per class, with group sizes beside every figure. Never used for model selection or the review threshold.
- **Alternatives:** no subgroup analysis; per-class subgroups; a different flag (`other_class_any`, `other_class_ge25`).
- **Reason:** 772 of 6211 train crops (12.4%) contain at least half of an other-class box, so the classifier could learn from surrounding damage instead of the labelled one. Fixed before any results.
- **Decided by:** me

### D-020 · Scope and schedule cut
- **Date:** 2026-10-27
- **Choice:**
  - Tue 27 Oct: finish A3, then A4 and A5 (augmentations: horizontal flip and brightness only).
  - Wed 28 Oct: A6 with epoch timing, the training runs, A11 to A13 (no Grad-CAM).
  - Thu 29 Oct: Phase C. Phase B only if Phase A is fully done by Wednesday night; otherwise it is documented as future work.
  - Cut: Phase D, Grad-CAM and Mask R-CNN. The ViT run is a linear probe or 1 seed (replaces the ViT part of D-010). The seed count for the ResNet runs is decided from the A6 epoch timing (D-010).
  - Working method: low-risk steps (code, tests, commit) are batched into one approval; work stops only for decisions, training starts and the test evaluation.
- **Alternatives:** the original schedule (Phase A by 25 Oct, Phase B by 27 Oct, Phase C on 28 Oct).
- **Reason:** It is Tuesday evening and the application closes on Friday 30 October. Scope is cut; the evaluation is not.
- **Decided by:** me

### D-021 · A5 augmentation: horizontal flip and brightness, train only
- **Date:** 2026-10-27
- **Choice:** Train crops only, before padding, with OpenCV:
  - Horizontal flip, p = 0.5.
  - Brightness multiplier ~ Uniform[0.8, 1.2], clipped to 0 to 255.
  - Val and test: no augmentation. Small rotations from the original brief are cut (D-020).
- **Alternatives:** vertical flip; stronger brightness changes; small rotations.
- **Reason:**
  - Flip: damage has no left/right meaning and cars are roughly symmetric. No vertical flip: photos have a consistent up/down.
  - Brightness: mimics lighting differences between sites and times of day. Kept at ±20% because light is the largest colour group (1298 of 2816 train photos, 46.1%, `outputs/metrics/data_verification.json`), and stronger brightening clips white paint and erases light scratches.
- **Grid check (2026-10-27):** the augmentation grid showed light-coloured crops at ×1.2 still clear, but the sample had no light scratch on a white car, so the check is reassuring, not conclusive.
- **Decided by:** me

### D-022 · A6 training settings
- **Date:** 2026-10-27
- **Choice:**
  - Class imbalance: class-weighted cross-entropy, weights = inverse train class frequency, normalised to mean 1, computed from the frozen train split. Same loss for every run.
  - Model selection: the checkpoint from the epoch with the best validation macro F1; ties go to the earlier epoch. Validation is used only for this and for A11.
  - Optimiser: AdamW, weight decay 1e-4. Head learning rate 1e-3; fine-tune backbone 1e-4 (×0.1 the head); scratch 1e-3 for all layers.
  - Schedule: cosine decay, no warmup.
  - Batch size: 64 for every run, so the comparison is like-for-like.
  - Mixed precision (AMP): on.
  - ViT: torchvision ViT-B/16 (ImageNet weights) as a linear probe (D-020). In the results it is compared with the frozen ResNet-18, since both keep their backbone frozen.
  - Seed count and epoch budget: decided after the A6 timing.
- **Alternatives:** plain cross-entropy; a weighted sampler; last-epoch checkpoint; SGD; constant or step schedule; other batch sizes; AMP off; ViT full fine-tune.
- **Reason:** Macro F1 is the headline metric, so the loss should not let scratch and dent dominate. One batch size for every run keeps the comparison like-for-like.
- **Implementation details (AI proposed; items marked * changed by me):** validation loss uses the same class weights; frozen backbones stay in eval mode during training, so BatchNorm statistics do not change; cosine decay is stepped every batch down to 0; the best checkpoint (full state dict) is saved to Drive on every improvement; the timing run prints no validation scores; each DataLoader worker gets its own seeded augmentation generator; *a real run refuses uncommitted changes to tracked files, while untracked run outputs do not block it; *`metrics.json` is marked finished only when a run completes, and only finished runs (local or Drive copy) are refused, so an interrupted run can be restarted from epoch 1; *after every epoch the run folder (`metrics.json`, `val_probs.csv`) is copied to `MyDrive/vdv/runs/<run>/`.
- **Decided by:** me
