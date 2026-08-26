# Agent Operating Procedures for Training Offload

You are an AI coding agent running on a host machine. Your task is to successfully execute this YOLO11n training pipeline from end to end on behalf of the remote author, returning a packaged result zip.

## Core Directives (CRITICAL)

1. **DO NOT modify the codebase.** You MUST NOT edit any `.ps1`, `.bat`, `.py`, or `.yaml` files. The pipeline is an exact, pre-agreed experimental recipe. "Fixing" a script to bypass an error (e.g., lowering resolution, skipping an export, disabling an assertion) silently corrupts the deliverables. 
2. **Remediate the environment, not the code.** If a script fails, the problem is in the host environment (missing dependencies, old drivers, locked files, path lengths, insufficient disk space). Use your autonomy to fix the *environment* so the script passes as written.
3. **Idempotency.** All `.ps1` scripts are designed to be safely re-run. If something crashes, fix the environment and run the script again.

## Execution Sequence

Execute the PowerShell scripts in `scripts/` sequentially. You can use the `.bat` launchers in the root or call the `.ps1` scripts directly (with `-ExecutionPolicy Bypass`).

### `scripts/1_setup.ps1`
**Your primary value is here.** This script creates the `.venv`, installs CUDA-enabled PyTorch + YOLO tools, and asserts GPU readiness.
- Run it. If it succeeds (`[OK] CUDA_OK...`), proceed.
- **If it fails:** Autonomously diagnose the environment. 
  - If Python 3.10-3.13 is missing and the script's `winget` fallback fails, install it yourself.
  - If PyTorch complains about CUDA, the NVIDIA driver is likely too old (< 580). Instruct the user to update it at nvidia.com, wait for them to do so, and retry.
  - Ensure the `.venv` is healthy and `torch.cuda.is_available()` is `True`.

### `scripts/2_download_data.ps1`
- Downloads and extracts ~2.5 GB. Ensure it completes and `datasets/usable/` is populated.

### `scripts/3_prepare_dataset.ps1`
- Converts the data. Should take minutes.

### (Optional) `scripts/4_train.ps1 -SmokeTest`
- Run this first to prove the GPU/CUDA/VRAM stack works end-to-end on a tiny subset of data.

### `scripts/4_train.ps1`
- **This will take several hours.** 
- Launch it and verify epochs are actively progressing. 
- Ensure the host machine is set to never sleep while plugged in (you may check/adjust powercfg if permitted, or instruct the user).
- If it crashes mid-way (e.g. thermal shutdown), re-running it restarts training from scratch. Do not edit the script to hack in resume support.

### `scripts/5_package_results.ps1`
- Runs NCNN/ONNX exports and model evaluation. 
- Produces `training-results-<date>.zip` in the repo root.
- Some export steps (like int8) are soft-failures (they warn and continue). This is expected. If the script finishes successfully, the package is ready.

## Final Handoff
Once step 5 completes and the `.zip` is generated, present a clear message to the human host instructing them to send the `.zip` file back to the author. Your job is then complete.
