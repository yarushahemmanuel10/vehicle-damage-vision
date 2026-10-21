# Corrections log

Every time something the AI assistant produced was wrong and was caught and fixed.

---

### C-001 · `.gitignore` silently excluded source code
- **Date:** 2026-10-21
- **What was wrong:** The rule `data/` matched every folder named `data`, including the code package `src/vdv/data/`. It would have been left out of the repo, breaking imports on Colab.
- **How it was caught:** Reviewing `git status --short --untracked-files=all` before the first commit; two expected files were missing.
- **Fix:** Anchored the rule to the repo root: `/data/`.
- **Lesson:** Check what git will actually commit; don't trust the ignore file by reading it.
