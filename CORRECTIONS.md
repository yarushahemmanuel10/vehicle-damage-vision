# Corrections log

Every time something the AI assistant produced was wrong and was caught and fixed.

---

### C-001 · `.gitignore` silently excluded source code
- **Date:** 2026-10-21
- **What was wrong:** The rule `data/` matched every folder named `data`, including the code package `src/vdv/data/`. It would have been left out of the repo, breaking imports on Colab.
- **How it was caught:** Reviewing `git status --short --untracked-files=all` before the first commit; two expected files were missing.
- **Fix:** Anchored the rule to the repo root: `/data/`.
- **Lesson:** Check what git will actually commit; don't trust the ignore file by reading it.

### C-002 · Per-session install could replace Colab's PyTorch
- **Date:** 2026-10-22
- **What was wrong:** The S4 draft installed all of `requirements.txt` every session, including `torch==2.11.0` and `torchvision==0.26.0`. Colab's builds are `2.11.0+cu130` / `0.26.0+cu130`. They match today only because pip ignores the `+cu130` label; if Colab moved to a newer PyTorch, pip would download 2.11.0 and replace the CUDA build every session.
- **How it was caught:** I checked and reviewed each issue and can explain it in my own words.
- **Fix:** torch/torchvision are excluded from the per-session install. The cell checks the installed build against the tested one and stops on a mismatch, and a constraints file prevents other packages from changing them.

### C-003 · Unzip was not safe to re-run
- **Date:** 2026-10-22
- **What was wrong:** `unzip` without `-o` asks before overwriting existing files. After an interrupted unzip, a re-run would wait for an answer that cannot be typed in a non-interactive command.
- **How it was caught:** I checked and reviewed each issue and can explain it in my own words.
- **Fix:** `unzip -o` (overwrite without asking).

### C-004 · A failed clone or pull did not stop the cell
- **Date:** 2026-10-22
- **What was wrong:** The git helper printed errors but the exit code was ignored. A failed clone would crash later with a confusing error; a failed pull would continue silently with out-of-date code.
- **How it was caught:** I checked and reviewed each issue and can explain it in my own words.
- **Fix:** Clone and pull raise an error immediately on a non-zero exit code.

### C-005 · Requirements expanded into the pip command line
- **Date:** 2026-10-22
- **What was wrong:** The S4 draft pasted the filtered requirements straight into the `%pip` command line, and filtered on the unstripped line, so an indented `torch` line or comment could slip through the filter. Expanding a list into the command line is also harder to read and reproduce.
- **How it was caught:** I checked and reviewed each issue and can explain it in my own words.
- **Fix:** The filtered list is written to `/content/reqs_session.txt`, filtering on the stripped line and the parsed package name, and installed with `pip install -r`.
