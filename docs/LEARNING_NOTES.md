# Learning notes

Plain-English explanations of each new concept, written as it comes up. Revise from this before interviews.

---

**CPU vs CUDA builds of PyTorch.** The suffix in `torch.__version__` shows the build: `+cpu` cannot use a GPU; `+cu130` is built for NVIDIA's CUDA 13.0. Colab installs the build that matches the runtime type, so the fix for "no GPU" is changing the runtime, not reinstalling PyTorch.

**Why the GPU matters.** Training repeats huge numbers of matrix multiplications. A GPU runs thousands of them in parallel, so fine-tuning that takes hours on a CPU takes minutes on a T4.

**Ephemeral vs persistent storage.** Colab's `/content` disk is wiped when the session ends. Google Drive, mounted at `/content/drive`, persists. So code lives on GitHub, data and model weights on Drive, and Colab is just the machine that runs them.

**`.gitignore` patterns.** `data/` matches a folder named `data` at any depth; `/data/` matches only the one at the repo root. Always check `git status` to see what will really be committed.

**Least-privilege tokens.** The GitHub token can only read and write this one repo and expires in 30 days. It lives in Colab Secrets, not in the code, so it never appears in the notebook or git history.
