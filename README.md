# Rescue Swarm Drone Human Detector — Training & Evaluation Suite

A self-contained, automated machine learning suite to train, benchmark, optimize, and package edge AI models (**YOLO11** and **YOLO26**) specialized in detecting **humans in drone aerial footage** (VisDrone2019-DET).

Designed for offloaded workstation training (e.g. laptop/desktop with NVIDIA RTX GPU) and edge deployment onto **Raspberry Pi 5** (ARM Cortex-A76) via **NCNN** (fp16/int8) and **ONNX**.

---

## ⚡ Quick Start: Choose Your Workflow

Choose the workflow that best fits your workflow and technical comfort:

```
                  ┌──────────────────────────────────────────────────┐
                  │           Rescue Swarm Training Suite            │
                  └─────────┬──────────────────┬─────────────────┬───┘
                            │                  │                 │
              Option A: Interactive   Option B: Autonomous    Option C: Classic
                     TUI                   Tournament            5-Step Batch
                            │                  │                 │
                      [ TUI.bat ]     [ Tournament.bat ]     [ 1-Install.bat ]
                            │                  │             [ 2-Download... ]
                            ▼                  ▼             [ 3-Prepare...  ]
                    Live Telemetry,      Multi-Phase Model   [ 4-Train.bat   ]
                    RPi 5 Sim, Checks    Sweep & Leaderboard [ 5-Package...  ]
```

### Option A: Unified Terminal Dashboard (TUI) — *Recommended*
Launch the interactive terminal workstation with real-time GPU/CPU telemetry, dataset health diagnostics, training controls, and Raspberry Pi 5 simulation.
- **Windows**: Double-click `TUI.bat` (or `windows/Pipeline-TUI.bat`)
- **Linux**: Run `python training/tui.py` inside your virtual environment

### Option B: Overnight Model Tournament — *Best for Model Selection*
Runs an autonomous multi-phase tournament (successive halving) testing multiple architectures, scales (nano, small, medium), and aerial-specific augmentations to find the optimal human detector.
- **Windows**: Double-click `windows/Tournament.bat` (headless) or `windows/Tournament-TUI.bat` (interactive)
- **Linux**: Run `./linux/Tournament.sh`

### Option C: Classic 5-Step Guided Pipeline — *Simple & Sequential*
Double-click each numbered batch file in `windows/` (or run shell scripts in `linux/`) in order:
1. `1-Install` ➔ 2. `2-Download-Dataset` ➔ 3. `3-Prepare-Dataset` ➔ *(Optional `QUICK-TEST`)* ➔ 4. `4-Train` ➔ 5. `5-Package-Results`

---

## 📋 System Requirements

| Requirement | Minimum / Recommended Specification |
|---|---|
| **Operating System** | Windows 10/11 64-bit or Linux (Ubuntu 22.04+, Arch, etc.) |
| **GPU** | NVIDIA GPU with CUDA support (RTX 40-series / 30-series / 20-series, GTX 16-series) |
| **NVIDIA Driver** | Driver version **580+** recommended for CUDA 12.x support |
| **Disk Space** | ~20 GB free space on local drive (dataset + venv + model runs) |
| **Network** | Internet connection for initial setup & VisDrone dataset download (~2.5 GB) |
| **Power & Sleep** | Laptop **must be plugged in**. Prevent sleep during training. |

> **⚠️ Important Power Setting:** Keep your machine awake during training:  
> *Windows Settings ➔ System ➔ Power ➔ Screen and sleep ➔ When plugged in, put my device to sleep: **Never***  
> *(Note: The tournament runner includes automatic Windows Sleep Inhibition).*

---

## 🚀 The 5-Step Training Pipeline

### Step 1: Environment Setup (`1-Install.bat` / `1-Install.sh`)
- Automated `.venv` virtual environment provisioning (Python 3.10–3.13).
- Installs PyTorch with CUDA acceleration, Ultralytics YOLO, OpenCV, ONNX, and NCNN tooling.
- Asserts NVIDIA driver and GPU readiness (`[OK] CUDA_OK <GPU_NAME>`).

### Step 2: Download Dataset (`2-Download-Dataset.bat` / `2-Download-Dataset.sh`)
- Downloads the official **VisDrone2019-DET** training and validation archives (~2.5 GB).
- Automatically unpacks raw images and annotations into `datasets/usable/`.

### Step 3: Dataset Preparation (`3-Prepare-Dataset.bat` / `3-Prepare-Dataset.sh`)
- Converts VisDrone bounding boxes into standard YOLO single-class format (`0: human`, merging pedestrians and people while ignoring ignored regions and vehicle classes).
- Generates train/validation splits and generates `datasets/usable/yolo-human/data.yaml`.

### Optional: Smoke Test (`QUICK-TEST.bat` / `QUICK-TEST.sh`)
- Runs a rapid 2-minute validation pass on a micro-subset of data to verify CUDA kernel execution, loss calculation, and VRAM stability before launching full training.

### Step 4: Model Training (`4-Train.bat` / `4-Train.sh`)
- Trains a two-stage YOLO human detector (Stage 1: frozen backbone for feature transfer; Stage 2: full network fine-tuning with cosine learning rate schedule).
- Saves training checkpoints, batch prediction samples, and loss curves under `runs/detect/train/`.

### Step 5: Package Deliverables (`5-Package-Results.bat` / `5-Package-Results.sh`)
- **Export & Quantization**: Converts the best checkpoint (`best.pt`) to:
  - `ncnn-fp16` & `ncnn-int8` (optimized for ARM NEON on Raspberry Pi 5)
  - `onnx-int8` / `onnx-fp32` (for ONNX Runtime)
- **Benchmarking**: Measures simulated Raspberry Pi 5 CPU throughput (4-thread throttling) and validation mAP.
- **Packaging**: Bundles model weights, export binaries, validation curves, prediction previews, and summary metrics into `training-results-<date>.zip`.

---

## 🏆 Overnight Model Tournament Framework

The tournament system (`scripts/tournament.ps1`, `training/tournament_runner.py`) automates overnight model exploration:

```
 Phase 1: Screening (640px)      Phase 2: Shortlist (960px)       Phase 3: Finalists (1280px)
 ┌───────────────────────────┐   ┌───────────────────────────┐   ┌───────────────────────────┐
 │ 8 Candidates              │   │ Top 4 Candidates          │   │ Top 2 Finalists           │
 │ (YOLO11n/s/m, YOLO26n,    │──▶│ Full unfreeze,            │──▶│ High-res refinement,      │
 │  drone augmentations)     │   │ higher resolution         │   │ fine-grained tuning       │
 └───────────────────────────┘   └───────────────────────────┘   └─────────────┬─────────────┘
                                                                               │
                                                                 ┌─────────────▼─────────────┐
                                                                 │ LEADERBOARD.md & Artifacts│
                                                                 │ Winner Model Checkpoint   │
                                                                 └───────────────────────────┘
```

### Key Features
- **VRAM Protection for RTX GPUs**: Dynamic batch size calculation preventing out-of-memory (OOM) faults on 4 GB – 8 GB GPUs (e.g. RTX 3050, RTX 4050).
- **Drone-Specific Augmentations**:
  - `drone_aerial`: Rotation invariance, oblique perspective tilt, scale jitter.
  - `small_object`: High box gain, distribution focal loss (DFL) boosting, early mosaic close for tiny distant targets.
- **Hard Validation Subset**: Evaluates candidate models against small/crowded aerial frames (`hard_val_subset`).
- **Composite Drone Curation Scoring**:
  $$\text{Score} = 0.35 \times \text{Recall} + 0.25 \times \text{Small Recall} + 0.20 \times \text{mAP}_{50\text{-}95} + 0.15 \times \text{Precision} + 0.05 \times \text{Hard Set}$$
- **Tournament Presets**:
  - `overnight_standard` (~6.5 hrs) — Default overnight sweep
  - `overnight_extended` (~8.0 hrs) — Extended fine-tuning budget
  - `fast_screening` (~2.5 hrs) — Rapid architecture comparison
  - `smoke_test` (~10-15 mins) — End-to-end pipeline verification

---

## 🥧 Raspberry Pi 5 Deployment & Simulation

This suite includes dedicated evaluation tooling (`training/eval/evaluate.py`) that simulates the Raspberry Pi 5 environment on your host machine:
- **ARM Cortex-A76 Mimicry**: Restricts execution to 4 CPU threads with an empirical frequency scaling correction factor (`--rpi-factor 0.55`).
- **NCNN Runtime**: NCNN provides ARM NEON acceleration and achieves **15–25+ FPS** on a physical Raspberry Pi 5 at 416px–512px resolution.
- **INT8 Quantization**: Calibration against the validation split produces minimal accuracy degradation while slashing memory footprint.

---

## 🛠️ Troubleshooting

| Problem | Symptom / Error | Solution |
|---|---|---|
| **Windows SmartScreen** | Blue "Windows protected your PC" popup | Click **More info** ➔ **Run anyway** |
| **Driver Outdated** | `driver ... is too old ... needs 580+` | Download & install the latest NVIDIA driver from [nvidia.com/drivers](https://www.nvidia.com/drivers), reboot, and re-run Step 1. |
| **No CUDA Acceleration** | `PyTorch CANNOT use your GPU` | Ensure NVIDIA driver is installed and reboot. `1_setup.ps1` / `1_setup.sh` installs CUDA-enabled PyTorch automatically. |
| **Path / OneDrive Issues** | Script fails finding files or access denied | Keep project folder on a local drive path (e.g. `C:\Dev\training-offload` or `~/training-offload`), avoiding OneDrive-synced folders. |
| **Low Free Disk Space** | Extraction or install crashes with no space | Free up ~20 GB on drive, delete `.venv` if corrupt, and restart from Step 1. |
| **Antivirus Lock** | Slow installation or locked files | Temporarily pause active antivirus scanning during Step 1 library installation. |

---

## 📁 Repository Structure

```
training-offload/
├── TUI.bat                      # Root one-click launcher for Pipeline TUI
├── requirements.txt             # Core Python package dependencies
├── AGENTS.md                    # Autonomous agent execution instructions
├── README.md                    # Project documentation
│
├── windows/                     # Windows Batch Launchers (Double-Click)
│   ├── 1-Install.bat            # Step 1: Venv & CUDA setup
│   ├── 2-Download-Dataset.bat   # Step 2: Download VisDrone archive
│   ├── 3-Prepare-Dataset.bat    # Step 3: Convert annotations to YOLO
│   ├── 4-Train.bat              # Step 4: Full training pipeline
│   ├── 5-Package-Results.bat    # Step 5: Export, benchmark, & zip
│   ├── QUICK-TEST.bat           # Smoke test launcher
│   ├── Pipeline-TUI.bat         # Unified pipeline TUI launcher
│   ├── Tournament.bat           # Headless overnight tournament launcher
│   └── Tournament-TUI.bat       # Interactive tournament TUI launcher
│
├── linux/                       # Linux Shell Launchers
│   ├── 1-Install.sh             # Step 1 setup
│   ├── 2-Download-Dataset.sh    # Step 2 download
│   ├── 3-Prepare-Dataset.sh     # Step 3 preparation
│   ├── 4-Train.sh               # Step 4 training
│   ├── 5-Package-Results.sh     # Step 5 packaging
│   ├── QUICK-TEST.sh            # Quick verification
│   └── Tournament.sh            # Tournament launcher
│
├── scripts/                     # Core Automation Scripts (PowerShell & Bash)
│   ├── 1_setup.ps1 / .sh        # Environment initialization
│   ├── 2_download_data.ps1 / .sh# Dataset download and extraction
│   ├── 3_prepare_dataset.ps1/.sh# Dataset conversion runner
│   ├── 4_train.ps1 / .sh        # Model training runner
│   ├── 5_package_results.ps1/.sh# Export, benchmark, & zip creator
│   ├── tournament.ps1 / .sh     # Tournament runner wrapper
│   ├── package_helpers.py       # Helper functions for packaging & zipping
│   └── _common.ps1 / .sh        # Shared CLI utility functions
│
├── training/                    # Machine Learning Codebase
│   ├── tui.py                   # Unified Interactive TUI Workstation
│   ├── train.py                 # YOLO training engine (two-stage recipe)
│   ├── prepare_dataset.py       # VisDrone-to-YOLO conversion & filtering
│   ├── export.py                # NCNN / ONNX export & INT8 calibration
│   ├── tournament_runner.py     # Tournament CLI & execution entrypoint
│   ├── eval/                    # Evaluation & RPi 5 Simulation
│   │   └── evaluate.py          # Latency & mAP evaluation with CPU throttling
│   └── tournament/              # Overnight Tournament Engine
│       ├── config.py            # Presets, candidate spaces, VRAM profiles
│       ├── controller.py        # Successive-halving phased scheduler
│       ├── metrics.py           # Evaluation & small-object scoring logic
│       ├── hard_dataset.py      # Hard validation subset builder
│       ├── platform_utils.py    # Windows sleep inhibitor & VT100 console
│       ├── reporter.py          # Markdown, CSV, JSON, and plot report generator
│       └── tui_tournament.py    # Dedicated tournament TUI monitor
│
├── tests/                       # Unit & Integration Tests
│   └── test_tournament.py       # Tournament framework test suite
│
└── datasets/ runs/              # Generated at runtime (dataset & training outputs)
```

---

## 📦 Deliverables & Hand-Off

Upon completing **Step 5** (`5-Package-Results.bat` / `5-Package-Results.sh`) or finishing the **Tournament**, the final package is generated in the repository root:
- `training-results-<timestamp>.zip`

Send this single zip file back to the project author. It contains:
- `weights/` (`best.pt`, `last.pt`, `best_ncnn_model/`, `best.onnx`)
- `benchmarks/` (mAP metrics, Raspberry Pi 5 simulated FPS & latency profiles)
- `plots/` (Precision-Recall curves, confusion matrices, batch predictions)
- `RESULTS.txt` / `LEADERBOARD.md` (Executive performance summary)
