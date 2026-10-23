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
