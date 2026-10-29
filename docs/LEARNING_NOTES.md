# Learning notes

Plain-English explanations of each new concept, written as it comes up. Revise from this before interviews.

---

**CPU vs CUDA builds of PyTorch.** The suffix in `torch.__version__` shows the build: `+cpu` cannot use a GPU; `+cu130` is built for NVIDIA's CUDA 13.0. Colab installs the build that matches the runtime type, so the fix for "no GPU" is changing the runtime, not reinstalling PyTorch.

**Why the GPU matters.** Training repeats huge numbers of matrix multiplications. A GPU runs thousands of them in parallel, so fine-tuning that takes hours on a CPU takes minutes on a T4.

**Ephemeral vs persistent storage.** Colab's `/content` disk is wiped when the session ends. Google Drive, mounted at `/content/drive`, persists. So code lives on GitHub, data and model weights on Drive, and Colab is just the machine that runs them.

**`.gitignore` patterns.** `data/` matches a folder named `data` at any depth; `/data/` matches only the one at the repo root. Always check `git status` to see what will really be committed.

**Least-privilege tokens.** The GitHub token can only read and write this one repo and expires in 30 days. It lives in Colab Secrets, not in the code, so it never appears in the notebook or git history.

**Keeping a token out of git commands (`GIT_ASKPASS`).** When git needs a password and cannot ask interactively, it runs the program named in `GIT_ASKPASS` and uses what it prints. Our script prints the token from an environment variable that exists only for the one push command, so the token never appears in a URL, a command, `.git/config` or any output.

**Pinning versions.** `requirements.txt` lists exact versions (`==`) taken from what was actually installed and tested, so the project can be rebuilt identically later. A pin like `torch==2.11.0` also matches the CUDA build `2.11.0+cu130`, because the part after `+` (a "local version label") is ignored when the pin does not mention it.

**Transitive dependencies.** Packages need other packages: imagehash needs PyWavelets and scipy. Pinning these too means a future install cannot quietly pull in different versions.

**Editable install (`pip install -e .`).** This installs our own `src/vdv` package by linking to the source folder rather than copying it, so code changes are picked up without reinstalling. Any script can then write `from vdv.data import ...` regardless of which folder it runs from.

**Constraints file (`pip install -c`).** A constraints file says "if anything needs this package, it must be exactly this version", but never asks pip to install anything. We use it to stop any other package from swapping out Colab's tested PyTorch build.

**Failing loudly.** The start-of-session cell stops with an error if `git clone` or `git pull` fails, or if PyTorch is not the tested build. Carrying on silently could mean training on old code or a different environment without noticing.

**Random seeds.** Training uses randomness everywhere: starting weights, data order, augmentations. `set_seed()` fixes Python's, NumPy's and PyTorch's random generators, so differences between runs come from the method, not from luck.

**Deterministic mode.** Even with seeds fixed, some GPU operations pick the fastest algorithm each time, which can change results slightly. `deterministic=True` reduces this run-to-run variation (it does not guarantee identical results) and makes training slower.

**pytest.** Any function whose name starts with `test_` in the `tests/` folder runs automatically with `python -m pytest`. A test passes if its `assert` statements hold.

**COCO box format.** COCO stores each box as `[x, y, width, height]`: the top-left corner plus size, in pixels. torchvision's detectors expect `[x1, y1, x2, y2]` (two corners), so boxes must be converted before detection training.

**COCO masks.** Each annotation can also carry a `segmentation`: either polygons (lists of x, y points; at least 3 points to form a shape) or RLE, a compressed pixel mask. Mask R-CNN trains on these.

**EXIF orientation.** Photos can store pixels sideways with a tag telling viewers to rotate them. OpenCV's default `imread` applies the tag; some annotation tools ignore it. If the two disagree, boxes land in the wrong place. CarDD has no such tags, so this cannot happen here.

**Fixing subgroups before results.** Grouping categories (such as car colours) after seeing results makes it easy, even unintentionally, to pick groups that tell a nicer story. Fixing the bins first keeps the subgroup analysis honest.

**Box area vs mask area.** A box is the smallest rectangle around a damage; the mask is the damage's actual shape. A thin diagonal crack can have a large box but a tiny mask. pycocotools' "AP small / medium / large" uses the mask area (the annotation's `area` field), so many cracks count as "small" even when their boxes are not.

**Aspect ratio.** Long side divided by short side: 1 is square, 10 is ten times longer than wide. Networks need a fixed square input, so very elongated crops must be stretched (distorts the shape), padded (wastes most of the input) or expanded with real context.

**Interpolation.** Resizing invents or merges pixels. `INTER_AREA` averages pixels and is best for shrinking; `INTER_LINEAR` blends neighbours and is the usual choice for enlarging. Enlarging a tiny crop cannot add detail: it only makes it blurrier.

**BGR vs RGB.** OpenCV loads colour images with the channels in the order blue, green, red (BGR); PyTorch, torchvision and pre-trained models expect red, green, blue (RGB). Pre-trained weights were learned with ImageNet's per-channel RGB mean and standard deviation, so a BGR image normalised with RGB statistics feeds the model swapped colours: training may still run, but transfer learning works worse, and a red tail light can look blue to the model. The A6 data loader converts BGR to RGB (`cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`) before ImageNet normalisation, and a test checks it on a synthetic pure-red image.

**Code-hash versioning.** The crop cache's version includes a hash of the crop rules and the build script, so editing either one changes the version and the old cache is refused. It finds the build script relative to the package source, which relies on the editable install (`pip install -e .`).

**Perceptual hash (pHash).** A 64-bit fingerprint of an image's overall structure, taken from the low-frequency part of a cosine transform. Recompressed, resized or slightly brightened copies keep nearly the same fingerprint, so near-duplicates can be found without comparing pixels.

**Hamming distance.** The number of the 64 bits that differ between two fingerprints: 0 means identical, around 32 is typical for unrelated images. pHash sets each bit by comparing a value with the median of all 64, so every hash has exactly 32 ones and every distance is even.

**What pHash misses.** Cropping shifts the whole frame, which changes the fingerprint about as much as switching to a different photo, so cropped copies are not separable by threshold. That is why the threshold covers recompress/resize/brightness copies only, and pairs just above it were checked by eye. Mirrored copies are caught by also hashing each val/test photo flipped left to right.

**Freezing a split with a hash.** A SHA-256 hash is a fingerprint of a file's exact bytes: change one character and the hash changes completely. Storing the manifest's hash at the freeze, and checking it in the tests and the data loader, makes any later change to the split impossible to miss.

**Data augmentation.** Random, label-preserving changes to training images (here a mirror flip and a brightness change), so the model sees more variety than the raw data holds and relies less on incidental details. Only training data is augmented: validation and test must reflect unaltered photos, or the scores stop describing real use.

**Saturation (clipping).** Pixel values live in 0 to 255. Brightening a white panel by 20% would push values above 255; they are clipped to 255, which turns detail into flat white. That is why the brightness range is kept small for a dataset with many light-coloured cars.

**Class-weighted loss.** Cross-entropy where each class's errors are multiplied by a weight. With weights proportional to 1 / class frequency, a mistake on a rare class (tire flat) costs more than one on a common class (scratch), so the model cannot score well by favouring the common classes. It targets macro F1, which averages classes equally.

**Model selection on validation.** After every epoch the model is scored on validation; the epoch with the best validation macro F1 is kept. Validation therefore influences the final model, which is why the test set stays untouched until the end.

**AdamW and weight decay.** AdamW adapts the step size per parameter and applies weight decay separately from the gradient, gently pulling weights towards zero to reduce overfitting.

**Cosine learning-rate schedule.** The learning rate starts at its set value and decreases along a cosine curve to zero by the last step: big steps early, fine adjustments at the end.

**Mixed precision (AMP).** Most calculations run in 16-bit floats, which the T4's tensor cores process much faster; a gradient scaler prevents tiny gradients from rounding to zero.

**Frozen backbone, linear probe, fine-tuning.** Frozen: the pre-trained network is a fixed feature extractor and only the final layer learns. Linear probe: the same idea, used to test how good a model's features are as they stand. Fine-tuning: every layer learns, the pre-trained ones at a lower learning rate so their knowledge is adjusted rather than overwritten.

**BatchNorm in a frozen network.** BatchNorm layers keep running averages of their inputs, and update them whenever the network is in training mode, even if no weights change. A frozen backbone is therefore kept in evaluation mode while its head trains.

**DataLoader workers and seeds.** Data is loaded by several worker processes in parallel. Each worker starts as a copy of the dataset, so without separate seeding every worker would draw the same "random" augmentations. Each worker gets its own generator, derived from the seeded loader.

**Overfitting one batch.** A quick test of a training loop: on a single fixed batch, a working model and optimiser should drive the loss close to zero. If they cannot, something in the loop is broken.

**Surviving a disconnect.** Colab can drop a session at any time. Each run writes its results after every epoch, copies them to Drive, and marks itself finished only at the end, so an interrupted run is visible as unfinished and can simply be run again.

**The "scheduler.step() before optimizer.step()" warning.** With mixed precision, the GradScaler can skip the very first optimiser step while it calibrates its scale (the first gradients overflow in 16-bit). The scheduler still advances, so one learning-rate value out of roughly 2,000 is skipped. Harmless.

**Validation loss rising while accuracy stays flat.** Late in fine-tuning the training loss approached zero and validation loss rose, while validation macro F1 stayed level. The model was still mostly right but became more confident, including on its mistakes. Confident mistakes are what a review threshold (A11) has to catch, so this affects how well routing by confidence can work.

**Review threshold (human-in-the-loop).** Each crop's confidence is its top softmax probability. Crops below a cut-off go to a human; the rest are labelled automatically. Raising the cut-off sends more crops to review and makes the auto-labelled share more accurate: the business chooses the trade-off.

**Calibration and ECE.** A model is calibrated if, among predictions made with 90% confidence, about 90% are right. A reliability table groups predictions by confidence and compares each group's accuracy with its mean confidence. Expected calibration error (ECE) is the weighted average gap. An overconfident model has accuracy below confidence, which weakens confidence-based routing.

**Fixing a selection rule in advance.** Choosing a threshold by looking at results invites picking whatever looks best. Writing the rule down first, then letting a script apply it, keeps the choice honest.

**Bootstrap confidence interval.** Resample the test set with replacement many times, recompute the metric each time, and take the middle 95% of the results. The width shows how much the score could move with a different test sample of the same size.

**Resampling photos, not crops.** Crops from the same photo share lighting, car and camera, so they are not independent. Resampling whole photos keeps them together; resampling crops would pretend there is more independent evidence than there is, and give intervals that are too narrow.

**Paired bootstrap.** Two models scored on the same resamples: the interval of their difference. If it excludes zero, the gap is unlikely to be test-set noise. Pairing is more sensitive than comparing two separate intervals, because both models face the same hard and easy photos in each resample.

**Testing once.** The test set is used a single time, after every choice is fixed. Looking at test results and then changing anything would turn the test set into a second validation set, and the reported score would be optimistic.

**Confident errors cannot be caught by a threshold.** On test, the four most confident errors of the fine-tune model had confidence above 0.9997, so any review threshold below that auto-labels them. Routing by confidence only helps if confidence is meaningful. Calibration, for example temperature scaling (dividing the logits by one constant fitted on validation), is the natural next step; it is future work here.

**Faster R-CNN.** A two-stage detector. A backbone (ResNet-50 with a Feature Pyramid Network) turns the image into feature maps at several scales. The Region Proposal Network (RPN) slides over them and proposes boxes likely to contain an object, starting from anchor boxes. The RoI heads then classify each proposal and refine its coordinates.

**Anchor boxes.** Fixed reference boxes of several sizes and shapes placed all over the image. The RPN predicts, for each anchor, whether it contains an object and how to shift and stretch it to fit.

**IoU (intersection over union).** The overlap of a predicted box and a true box, divided by the area they cover together: 1 is a perfect match, 0 is no overlap. A detection counts as correct at IoU 0.5 if it overlaps the true box by at least half in this sense.

**Non-maximum suppression (NMS).** Detectors predict many overlapping boxes for one object. NMS keeps the highest-scoring box and removes others that overlap it too much (by IoU), so each object is reported once.

**mAP.** For each class, precision is averaged over recall levels (AP); mAP averages AP over classes. mAP@50 counts a detection as correct at IoU ≥ 0.5; mAP@50:95 averages over IoU thresholds 0.50 to 0.95, so it also rewards tight boxes. COCO's AP-small, -medium and -large split the objects by area (here the mask area): small is below 32 × 32 pixels.

**Flipping boxes with the image.** When an image is mirrored, every box must be mirrored too: the new left edge is the image width minus the old right edge. Forgetting this teaches the detector to look in the wrong place.

**Frozen BatchNorm in detectors.** Detection batches are small (4 images here), too small for reliable BatchNorm statistics, so torchvision keeps the backbone's BatchNorm layers fixed at their pre-trained values.

**Learning-rate warmup.** Starting at a small learning rate and increasing it linearly over the first few hundred steps, to avoid unstable early updates in large models trained with SGD.

## Reproduce validation before a one-off test run (B5)

A one-off test run has no second chance, so the script first re-runs validation with the saved checkpoint and checks it gets the same mAP as during training. If it does, the checkpoint, the label mapping and the evaluation code are proven to be the ones that produced the training numbers, before the test set is touched. Exact equality isn't expected: float16 (mixed-precision) inference on a GPU can change the last digits, so a small tolerance (0.002) is allowed.
