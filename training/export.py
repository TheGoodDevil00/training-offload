#!/usr/bin/env python3
"""
export.py
=========
Export / optimize a trained YOLO11n human-detector for deployment on the
Raspberry Pi 5 (ARM Cortex-A76).

The recommended deployment path for the Pi 5 is **NCNN** with int8 (or fp16)
quantization. NCNN is built around ARM NEON instructions and is the fastest
open runtime for this SoC; a YOLO11n exported to NCNN at imgsz=416 typically
reaches ~15-25 FPS on a Raspberry Pi 5.

Secondary option: **ONNX int8** (run with onnxruntime on the Pi).

Usage
-----
    # NCNN fp16 (fast enough for 15+ FPS, best accuracy retained):
    python training/export.py --model runs/detect/train/weights/best.pt --format ncnn --imgsz 416 --half

    # NCNN int8 (fastest / smallest; needs the export to run the calibration):
    python training/export.py --model runs/detect/train/weights/best.pt \
        --format ncnn --imgsz 416 --int8 --data datasets/usable/yolo-human/data.yaml

    # ONNX int8 (alternative, run with onnxruntime):
    python training/export.py --model runs/detect/train/weights/best.pt \
        --format onnx --imgsz 416 --int8 --data datasets/usable/yolo-human/data.yaml

The exports are written next to the model as <name>_ncnn_model / <name>.onnx.

NOTE: NCNN / onnx / onnxslim are NOT installed in this venv yet; ultralytics
will auto-install them at export time. Run this on the dev machine, then copy
the exported folder to the Pi.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Export YOLO11n for Raspberry Pi 5 deployment")
    p.add_argument("--model", type=str, default="runs/detect/train/weights/best.pt",
                   help="Trained weights (default: runs/detect/train/weights/best.pt)")
    p.add_argument("--format", type=str, default="ncnn", choices=["ncnn", "onnx"],
                   help="Export format. ncnn = recommended for RPi 5 (ARM NEON).")
    p.add_argument("--imgsz", type=int, default=416,
                   help="Inference input size; smaller = faster on the Pi")
    p.add_argument("--half", action="store_true", help="fp16 quantization (NCNN)")
    p.add_argument("--int8", action="store_true",
                   help="int8 quantization (needs --data for calibration)")
    p.add_argument("--data", type=str, default="datasets/usable/yolo-human/data.yaml",
                   help="data.yaml used to calibrate int8 quantization")
    p.add_argument("--dynamic", action="store_true", help="ONNX dynamic batch (optional)")
    p.add_argument("--simplify", action="store_true", help="ONNX graph simplification (optional)")
    return p.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        # Try common fallback paths if default or user path wasn't found directly
        fallbacks = [
            Path("runs/detect/train/weights/best.pt"),
            Path("runs/detect/runs/detect/train/weights/best.pt"),
            Path("yolo11n.pt"),
        ]
        found = None
        for fb in fallbacks:
            if fb.exists():
                found = fb
                break
        if found:
            print(f"[export] Warning: specified model '{args.model}' not found, falling back to '{found}'")
            model_path = found
        else:
            raise FileNotFoundError(
                f"Model weights file not found: '{args.model}'. "
                f"Please train a model first using `python training/train.py` or specify valid weights."
            )

    model = YOLO(str(model_path))
    if args.int8 and not args.data:
        print("[export] WARNING: int8 quantization wants --data to calibrate; "
              "continuing with default calibration set.")

    export_kwargs = {
        "format": args.format,
        "imgsz": args.imgsz,
        "half": args.half,
        "int8": args.int8,
    }
    if args.int8 and args.data:
        export_kwargs["data"] = args.data
    if args.dynamic:
        export_kwargs["dynamic"] = True
    if args.simplify and args.format == "onnx":
        export_kwargs["simplify"] = True

    out = model.export(**export_kwargs)
    print(f"\n[export] done -> {out}")

    if args.format == "ncnn":
        print("\nOn the Raspberry Pi, copy this folder and run:")
        print(f"  python training/inference.py --model {out} "
              f"--input drone.mp4 --output out.mp4 --imgsz {args.imgsz}")
        print(f"(keep --imgsz {args.imgsz} matching the export for best speed)")


if __name__ == "__main__":
    main()
