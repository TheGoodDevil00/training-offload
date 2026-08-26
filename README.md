# Drone Human Detector - Training Package

This folder trains a small AI model (YOLO11n) that detects **humans in drone
footage**. Everything is automated: you just double-click five files in order,
wait, and send one zip file back.

No programming knowledge needed. If a window shows red text, screenshot it and
send it back.

---

## Before you start (2 minutes)

You need:

| Requirement | Detail |
|---|---|
| Windows 10 or 11, 64-bit | |
| NVIDIA graphics card | RTX 20xx or newer recommended (GTX 16xx works too) |
| ~20 GB free disk space on C: | The dataset + software take about 15 GB |
| Internet | Downloads about 3 GB total (dataset + PyTorch) |
| Time | Setup ~30 min, training several hours (overnight is perfect) |

**Keep the laptop plugged in.** During the training step (step 4), also stop
Windows from sleeping:
`Settings > System > Power > Screen and sleep > When plugged in, put my device to sleep: Never`
(putting the *screen* to sleep is fine).

Keep this folder somewhere simple like `C:\Dev\training-offload` or your
Desktop. Avoid OneDrive-synced folders (Documents is often OneDrive-synced) -
they can lock files while training runs.

---

## The steps - run them in order, one at a time

Double-click each file, watch the black window until it says what to do next,
then close it (or it pauses and waits for a keypress).

### `1-Install.bat` — one-time setup (~10-30 min)
Checks your GPU, installs Python + all AI libraries if missing, verifies
everything works. You'll see `[OK] CUDA_OK <your GPU name>` when done.

### `2-Download-Dataset.bat` (~5-40 min depending on internet)
Downloads the VisDrone2019-DET drone dataset (~2.5 GB) and unpacks it.

### `3-Prepare-Dataset.bat` (~2-5 min)
Converts the raw annotations into training format (single class: human).

### Optional: `QUICK-TEST.bat` (~5-10 min)
A tiny 2-minute-scale training run just to prove everything works before the
real thing. Recommended once.

### `4-Train.bat` — THE LONG ONE (several hours; overnight recommended)
Trains in two stages (frozen backbone first, then full fine-tune):
100 + 40 epochs at 512 px. Just leave the window open. If the computer
crashes or the window closes, double-click it again - it restarts training
from scratch.

### `5-Package-Results.bat` (~15-30 min)
Exports the trained model (NCNN fp16/int8 + ONNX int8 for Raspberry Pi),
measures accuracy (mAP) and speed, renders prediction previews, and zips
everything up.

### Send back
The file `training-results-<date>.zip` in this folder. That's it!
(It contains the weights, exports, metrics, plots, and a RESULTS.txt summary.)

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Blue "Windows protected your PC" popup | Click **More info** → **Run anyway** |
| "No NVIDIA GPU driver detected" | Install/update the driver: https://www.nvidia.com/drivers then reboot |
| "driver ... is too old ... needs 580+" | Same as above - update the NVIDIA driver |
| "PyTorch ... CANNOT use your GPU" | Update driver from nvidia.com, reboot, run `1-Install.bat` again |
| Antivirus makes step 1 very slow | Temporarily pause it while installing (only step 1) |
| Ran out of disk space | Free up space on C:, delete the `.venv` folder, re-run step 1 |
| Any other red error text | Screenshot the whole window and send it |

---

## What's in here (for reference)

```
1-Install.bat ... 5-Package-Results.bat   <- the five steps (double-click these)
QUICK-TEST.bat                            <- optional mini-training sanity check
training\                                 <- the actual training scripts
scripts\                                  <- automation behind the .bat files
datasets\ runs\ output\                   <- created while running (can be deleted after)
```

Source pipeline: https://github.com/TheGoodDevil00/model-training
