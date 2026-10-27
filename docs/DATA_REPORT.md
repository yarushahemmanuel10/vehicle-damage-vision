# Data report

All numbers below are read from `outputs/metrics/data_verification.json`, produced by
`python scripts/verify_data.py --root <CarDD_COCO>`. Every zero-count statement is asserted
against that file before this page is generated.

## A1 · Verification

- **Images:** train 2816, val 810, test 374 (4000 total). Every image listed in the annotations is on disk, none are extra, all load with OpenCV, and every size matches the annotations.
- **Annotated damage instances:** 8740 (train 6211 / val 1744 / test 785). The CarDD paper (Wang, Li and Wu, *IEEE Transactions on Intelligent Transportation Systems*, 2023) describes "over 9,000"; this project uses the count from its own report.
- **Instances per category (all splits):** {'scratch': 3595, 'dent': 2543, 'crack': 898, 'lamp broken': 704, 'glass shatter': 681, 'tire flat': 319}
- **Masks:** 8740 of 8740 annotations have a non-empty segmentation mask.
- **Splits:** no file name appears in two splits; category IDs and names are identical in all three annotation files; no duplicate file names or image IDs within a split.
- **Boxes:** none with zero area and none outside the image (maximum overshoot 0.0 px), so no clipping or dropping is needed.
- **EXIF orientation:** no image carries an orientation tag, so images load in the orientation they were annotated in.
- **image_info.xlsx:** 4000 rows, one per image, 0 not matched to a split. Cross-check mismatches against the annotations: {'width': 0, 'height': 0, '#instances': 0, '#categories': 0}.

## For the Limitations section

- **98.0% of images are partial views** (close-ups of part of a car) rather than whole vehicles. Results may not transfer to full-vehicle photos.

## Box statistics (train only, D-013)

All numbers from `outputs/metrics/box_stats_train.json` (`python scripts/box_stats.py`).

- **Boxes:** 6211. Shorter side: median 174.0 px, 5th percentile 30.9 px, minimum 7.3 px.
- **Shorter side below 32 px:** 333 boxes, of which 174 are cracks (26.7% of train cracks).
- **Aspect ratio (long / short):** median 1.61, 95th percentile 4.6, maximum 25.66.
- **Overlap with another class:** 2354 boxes share pixels with a box of a different class; 478 are at least half covered by one.
- **COCO size buckets:** by box area {'small': 111, 'medium': 942, 'large': 5158}; by mask area {'small': 321, 'medium': 1424, 'large': 4466}.

## For the error analysis (observed on the train contact sheets)

- **Context risk:** full square-expansion of crack #1984 pulled a tail light into the crop, an object resembling another class (`lamp broken`). Context added by the margin, minimum window and aspect cap should be checked for similar effects.
- **Dent/scratch ambiguity:** some boxes labelled `dent` look like scratches (#1747, #1748, and long thin "dents" such as #355 and #433). Expect dent/scratch confusion; it may reflect genuinely ambiguous damage or label noise.

## A2 · Crops (version `v1-d432e94e`)

All numbers from `outputs/metrics/crops_summary.json` (`python scripts/build_crops.py`). Rules: D-014 margin 0.1, D-015 minimum window 64 px, D-016 aspect cap 3.0; stored with long side 224 px.

- **Crops:** 8740 (train 6211, val 1744, test 785), one per annotation.
- **Train, minimum window applied:** 736 crops, including 289 of 651 cracks (44.4%).
- **Train, aspect cap applied:** 598 crops; cap not reachable at the image edge: 0.
- **Train, another-class box inside the final window (share of that box inside the window):** any 2764 (44.5%), at least 25% 1306 (21.0%), at least 50% 772 (12.4%). Highest at 50% or more: lamp broken, 160 of 494 (32.4%).
- This contamination measure (share of the other box inside the window) is not comparable with the box-statistics overlap above (share of this box covered by another).
- **IDs:** annotation IDs are not unique across splits (1744 shared); image IDs are. The manifest is keyed on (split, ann_id).
- The manifest is provisional until A4 (after the A3 duplicate check).

## A3 · Near-duplicate check (D-006 amendment)

All numbers from `outputs/metrics/dup_calibration_train.json` and `outputs/metrics/dup_summary.json`.

- **Calibration (train only, 200 photos, seed 42):** distance from synthetic copies to their original: JPEG q70 max 2, half-size resize max 2, brightness ±10% max 4; 5 to 10% crop median 12.0, max 26; all combined median 10.0, max 26. Nearest different train photo: min 8, median 18.0, max 20.
- **Threshold:** distance ≤ 6 is a duplicate.
- **Automatic duplicates (pairs):** {'val-train': 4, 'test-train': 0, 'val-test': 0}. Photos removed: val 5, test 0. Crops excluded: {'train': 0, 'val': 8, 'test': 0}.
- **Spot-check by eye:** 11 pairs (including one train↔train pair, which never removes a photo): 1 marked duplicate, 10 kept.
- **Crops remaining for classification:** {'train': 6211, 'val': 1736, 'test': 785}.
- **Limitation:** pHash misses copies cropped beyond light trimming, and watermarks or overlays; only the closest pairs above the threshold were checked by eye.

### A3 spot-check finding

- 1 of the 10 pairs above the threshold that were checked by eye was a cropped duplicate that pHash missed (val/000254.jpg ↔ train/003104.jpg, distance 10); it is now removed.
- Val/test photos whose nearest match is at distance 8 to 12 (from `outputs/metrics/dup_candidates.json`): val↔train 33, test↔train 17, val↔test 3. Only the 10 closest pairs were checked by eye, so further cropped duplicates may remain among these.

## A4 · Frozen split

- `splits/crops_manifest.csv` frozen on 2026-10-27; its SHA-256 (`75c2afd5b7751df4…`) is stored in `splits/crops_manifest.sha256`. The tests and the data loader (`vdv.data.manifest.load_split`) refuse a manifest that differs from it.
- **Crops used for classification** (duplicates excluded): train 6211, val 1736, test 785.
- Label index = CarDD category ID − 1: dent 0, scratch 1, crack 2, glass shatter 3, lamp broken 4, tire flat 5.
- The test split is locked until the final evaluation (A12).

## A13 · Confident errors (test, fine-tune seed 42)

From `outputs/metrics/a13_examples.json` and `splits/crops_manifest.csv`. "Other-class max share" is the largest share of any other-class box that lies inside the crop's final window (D-014); `ge50` means at least half.

| Example | Test ann | True | Predicted | Confidence | Other-class max share | ge50 |
|---|---|---|---|---|---|---|
| #7 | 101 | crack | dent | 1.0000 | 0.00 | 0 |
| #8 | 249 | scratch | dent | 1.0000 | 0.00 | 0 |
| #9 | 163 | crack | scratch | 0.9999 | 0.35 | 0 |
| #10 | 379 | scratch | dent | 0.9997 | 0.35 | 0 |

- **0 of the 4** most confident errors contain at least half of a box of another damage type; **2 of 4** contain some part of one. All 4 were auto-labelled at confidence ≥ 0.9997.
- **My observation (private A13 figure):** #7 and #8 show a dominant dent around the labelled crack and scratch; #9 is a thin line on a dark panel; #10 shows both a dent and scratch marks; the two review cases show mixed damage.
- **My finding (visual inspection of a small sample):** in 2 of the 4 most confident errors, the crop visibly appears to contain dent damage that is not annotated as a separate box. This suggests some mixed damage is unlabelled, so the contamination flag likely underestimates mixed damage. Based on visual inspection of a small sample.
