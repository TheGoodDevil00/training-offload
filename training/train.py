#!/usr/bin/env python3
"""
train.py
========
Train a YOLO11n human-detector on the VisDrone-derived YOLO dataset produced
by `prepare_dataset.py`.

Typical usage
-------------
    # First convert the dataset:
    python training/prepare_dataset.py

    # Then train (uses the GPU if available, otherwise CPU).
    # batch=-1 auto-selects the largest batch that fits VRAM (recommended on
    # the 4 GB RTX 3050), freeze=10 keeps the COCO-pretrained backbone frozen
    # for a faster, higher-quality fine-tune, and single_cls is on because the
    # dataset has a single 'human' class:
    python training/train.py --epochs 100 --batch -1 --imgsz 512
                             --freeze 10 --single-cls

    # Two-stage recipe: freeze the backbone for the first N epochs, then
    # unfreeze (freeze=0) and fine-tune the whole net with cosine LR.
    # Stage 1:  python training/train.py --freeze 10 --epochs 40 --imgsz 512 --single-cls
    # Stage 2:  python training/train.py --freeze 0 --epochs 40 --imgsz 512 --single-cls --cos-lr \
    #                                     --weights runs/detect/train/weights/best.pt

    # Quick smoke test (validate the whole pipeline in a couple of minutes):
    python training/train.py --epochs 3 --fraction 0.2 --profile

    # Sweep the chosen inference input size for speed (smaller = faster on RPi):
    python training/train.py --imgsz 416 --epochs 100 --batch -1 --freeze 10 --single-cls

The model weights are saved to:
    runs/detect/train/weights/best.pt   (highest validation mAP)
    runs/detect/train/weights/last.pt   (last epoch)
"""

import argparse

import torch
import shutil
from pathlib import Path

from ultralytics import YOLO

DATASET_YAML = "datasets/usable/yolo-human/data.yaml"


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO11n human detector (drone view)")
    p.add_argument("--data", type=str, default=DATASET_YAML, help="Path to data.yaml")
    p.add_argument("--weights", type=str, default="yolo11n.pt",
                   help="Pretrained weights to start from (yolo11n.pt) or 'yolo11n.yaml' for from-scratch")
    p.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    p.add_argument("--imgsz", type=int, default=512, help="Training input size (px, divisible by 32)")
    p.add_argument("--batch", type=int, default=16,
                   help="Batch size (default 16 for optimal VRAM headroom on RTX 3050)")



    p.add_argument("--freeze", type=int, default=10,
                   help="Freeze the first N layers/backbone during training "
                        "(0 = freeze nothing). For a COCO-pretrained fine-tune, 10 "
                        "keeps the backbone frozen for a faster + higher-quality "
                        "transfer, then re-run with 0 to fine-tune the whole net.")
    p.add_argument("--single-cls", action=argparse.BooleanOptionalAction, default=True,
                   help="Single-class training (default on: dataset has one class)")
    p.add_argument("--fraction", type=float, default=1.0,
                   help="Fraction of the dataset to train on (e.g. 0.2 for a quick "
                        "smoke test)")
    p.add_argument("--profile", action="store_true",
                   help="Profile training to find the GPU vs data-loading bottleneck")
    p.add_argument("--cos-lr", action="store_true",
                   help="Use a cosine LR schedule (smoother convergence)")
    p.add_argument("--close-mosaic", type=int, default=10,
                   help="Disable mosaic augmentation for this many final epochs")
    p.add_argument("--channels-last", action="store_true",
                   help="Use channels_last memory format (small VRAM/speed win on CUDA)")
    p.add_argument("--multiscale", action="store_true",
                   help="Multi-scale training (raises VRAM; usually only if it fits)")
    p.add_argument("--device", type=str, default=None,
                   help="0,1,.. for GPU, 'cpu' for CPU (default: auto GPU)")
    p.add_argument("--patience", type=int, default=30,
                   help="Stop early if no val mAP improvement for N epochs")
    p.add_argument("--cache", choices=["True", "False", "ram", "disk"], default="False",
                   help="Cache images (ram/disk/False). Default False prevents OOM on systems without swap.")

    p.add_argument("--workers", type=int, default=8,
                   help="Data loader workers (default 8; raise toward CPU core count)")
    p.add_argument("--project", type=str, default="runs/detect", help="Output project folder")
    p.add_argument("--name", type=str, default="train", help="Experiment name")
    return p.parse_args()


def main():
    args = parse_args()

    # Auto-select device: GPU if available, else CPU.
    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[train] device = {device}  (cuda available: {torch.cuda.is_available()})")
    print(f"[train] epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} "
          f"freeze={args.freeze} single_cls={args.single_cls} fraction={args.fraction}")
    if args.freeze > 0:
        print("[train] backbone frozen (freeze=%d); re-run with --freeze 0 to "
              "fine-tune the whole net (two-stage recipe)." % args.freeze)
    if args.fraction < 1.0:
        print("[train] WARNING: using fraction=%.2f of the dataset (smoke test). "
              "Rerun with fraction=1.0 for the real run." % args.fraction)

    # Purge previous runs to ensure clean overwrite in runs/detect/train
    project_path = Path(args.project).resolve()
    target_run_dir = project_path / args.name
    if target_run_dir.exists():
        print(f"[train] Removing previous run directory: {target_run_dir}...")
        try:
            shutil.rmtree(target_run_dir)
        except Exception as e:
            print(f"[train] Warning removing {target_run_dir}: {e}")
    model = YOLO(args.weights)

    # Convert string boolean choices to actual booleans for ultralytics cache arg
    cache_arg = False if args.cache == "False" else (True if args.cache == "True" else args.cache)

    # amp is auto-enabled on CUDA; disable it if you hit numeric issues on CPU.
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,             # -1 = auto-select max batch that fits VRAM
        device=device,
        patience=args.patience,
        cache=cache_arg,
        project=str(project_path),
        name=args.name,
        exist_ok=True,               # overwrite runs/detect/train directly
        workers=args.workers,         # keep the dataloader feeding the GPU
        freeze=args.freeze,           # 10 = keep COCO-pretrained backbone frozen
        single_cls=args.single_cls,   # on by default (single 'human' class)
        fraction=args.fraction,       # <1.0 for a quick smoke test
        profile=args.profile,         # find GPU vs data-loading bottleneck
        cos_lr=args.cos_lr,           # cosine LR schedule
        close_mosaic=args.close_mosaic,
        channels_last=args.channels_last,
        multi_scale=args.multiscale,  # maps to ultralytics 'multi_scale' arg
        # ladder-tuning trick: train at 640 then fine-tune at the target size
        # --- you can lower 'conf' later at inference instead of re-training.
        val=True,
        plots=True,
    )

    print("\nTraining finished. Best weights:")
    print(f"  {args.project}/{args.name}/weights/best.pt")
    print("\nNext step:  python training/inference.py --model runs/detect/train/weights/best.pt")


if __name__ == "__main__":
    main()
