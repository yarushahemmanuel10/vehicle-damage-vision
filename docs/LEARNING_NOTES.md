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
