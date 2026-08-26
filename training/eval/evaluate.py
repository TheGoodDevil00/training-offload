#!/usr/bin/env python3
"""
evaluate.py
===========
Laptop-based evaluation suite for the human detector that recreates
Raspberry Pi 5 compute constraints so you can predict on-device performance
without a Pi.

Mimics of the Pi 5 (4 x Cortex-A76 @ ~2.4 GHz, 4 GB):
  * CPU-only inference (the Pi has no GPU acceleration for these models)
  * thread-count parity (locks Torch/OpenCV/NCNN to the Pi's core count)
  * a correction factor for A76's lower per-core speed vs a laptop core
    (--rpi-factor, default 0.55 => est. Pi 5 FPS = raw FPS * factor)
  * a RAM-limit alert (--mem-gb, default 4.0 GB)

Limitations: cannot replicate exact Pi clocks, cache, memory bandwidth or
thermal throttling, so treat the estimated Pi 5 FPS as a planning guide.

Subcommands
-----------
  speed      -> real FPS / latency on a video, swept over imgsz & formats
  accuracy   -> mAP50 / mAP50-95 on the val split (Ultralytics val())

Examples
--------
  # Throughput across formats & input sizes (the 15+ FPS check):
  python training/eval/evaluate.py speed \
      --models runs/detect/train/weights/best.pt \
               runs/detect/train/weights/best_ncnn_model \
      --video sample_drone.mp4 --imgsz 640,416,352

  # Accuracy at the deployment size:
  python training/eval/evaluate.py accuracy \
      --model runs/detect/train/weights/best.pt --imgsz 416
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import cv2


# --------------------------------------------------------------------------- #
# Raspberry Pi 5 constraint simulation
# --------------------------------------------------------------------------- #
def apply_rpi_constraints(threads: int, mem_gb: float):
    """Pin the runtime to CPU-only + a fixed core count (like the Pi 5)."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "NCNN_THREADS"):
        os.environ[var] = str(threads)
    os.environ["OMP_DYNAMIC"] = "false"  # keep the fixed thread count

    cv2.setNumThreads(threads)

    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(min(threads, 4))

    # Keep Torch from grabbing the GPU: we are simulating a CPU-only board.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print(f"[rpi-sim] CPU-only, threads={threads} "
          f"(Pi 5 has 4 Cortex-A76 cores), mem-limit={mem_gb:.1f} GB")

    try:
        import resource  # POSIX-only; absent on Windows

        def _rss_gb():
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    except ImportError:
        def _rss_gb():
            return 0.0  # RSS guard not available on Windows

    if mem_gb and _rss_gb() > mem_gb:
        print(f"[rpi-sim] WARNING: current RSS {_rss_gb():.2f} GB exceeds "
              f"Pi 5 limit {mem_gb:.1f} GB")
    return _rss_gb


# --------------------------------------------------------------------------- #
# Speed benchmark
# --------------------------------------------------------------------------- #
def load_frames(video: str, frames: int):
    """Read up to `frames` frames from a video into memory."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = []
    while len(buf) < frames:
        ok, frame = cap.read()
        if not ok:
            break
        buf.append(frame)
    cap.release()
    print(f"[speed] preloaded {len(buf)} frames from {video} @ {fps:.1f} fps "
          f"({w}x{h})")
    return buf


def benchmark_model(model, frames: list, imgsz: int, conf: float, iou: float,
                    warmup: int):
    """Run inference on preloaded frames; return {mean, p50, p90, fps}."""
    # Warm up (JIT / allocation)
    for f in frames[:warmup]:
        model(f, imgsz=imgsz, conf=conf, iou=iou, verbose=False)

    lat = []
    for f in frames:
        t0 = time.perf_counter()
        model(f, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
        lat.append((time.perf_counter() - t0) * 1000.0)  # ms

    mean_ms = statistics.mean(lat)
    return {
        "imgsz": imgsz,
        "mean_ms": round(mean_ms, 2),
        "p50_ms": round(statistics.median(lat), 2),
        "p90_ms": round(statistics.quantiles(lat, n=10)[8], 2),
        "fps": round(1000.0 / mean_ms, 1),
    }



# --------------------------------------------------------------------------- #
# Command implementations (receive a parsed namespace)
# --------------------------------------------------------------------------- #
def run_speed(a):
    apply_rpi_constraints(a.threads, a.mem_gb)
    from ultralytics import YOLO

    frames = load_frames(a.video, a.frames)
    imgsz_list = [int(s) for s in a.imgsz.split(",") if s.strip()]
    report = {"mode": "speed", "threads": a.threads,
              "rpi_factor": a.rpi_factor, "results": []}

    print(f"\n{'model':<36}{'imgsz':<7}{'fps':<8}{'pi5_fps':<9}"
          f"{'mean(ms)':<10}{'p50(ms)':<9}{'p90(ms)':<9}")
    for mpath in a.models:
        try:
            model = YOLO(mpath)
        except Exception as e:  # e.g. missing ncnn/onnx runtime
            print(f"  [skip] cannot load {mpath}: {e}")
            continue
        for imgsz in imgsz_list:
            r = benchmark_model(model, frames, imgsz, a.conf, a.iou, warmup=5)
            r["model"] = str(mpath)
            r["pi5_fps"] = round(r["fps"] * a.rpi_factor, 1)
            report["results"].append(r)
            print(f"{str(mpath):<36}{imgsz:<7}{r['fps']:<8}"
                  f"{r['pi5_fps']:<9}{r['mean_ms']:<10}{r['p50_ms']:<9}"
                  f"{r['p90_ms']:<9}", flush=True)

    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2))
        print(f"\nReport written to {a.out}")

    best = max(report["results"], key=lambda x: x["pi5_fps"], default=None)
    if best:
        hit = best["pi5_fps"] >= 15
        print(f"\nFastest: {best['model']} @ imgsz {best['imgsz']} "
              f"=> est Pi 5 FPS {best['pi5_fps']} "
              f"{'(>= 15 FPS target MET)' if hit else '(below 15 FPS target)'}")


def run_accuracy(a):
    apply_rpi_constraints(a.threads, a.mem_gb)
    from ultralytics import YOLO

    model = YOLO(a.model)
    print(f"\n[accuracy] validating {a.model} @ imgsz {a.imgsz}, CPU, "
          f"{a.threads} threads ...")
    metrics = model.val(data=a.data, imgsz=a.imgsz, device="cpu",
                        conf=a.conf, iou=a.iou, workers=a.threads)

    row = {
        "mode": "accuracy", "model": str(a.model), "imgsz": a.imgsz,
        "threads": a.threads,
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50-95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
    }
    print(f"\n{row}")
    if a.out:
        Path(a.out).write_text(json.dumps(row, indent=2))
        print(f"Report written to {a.out}")


def main():
    parser = argparse.ArgumentParser(prog="evaluate.py")

    sub = parser.add_subparsers(dest="cmd", required=True)

    def _shared(p):
        p.add_argument("--out", type=str, default=None,
                       help="Optional JSON report path")
        p.add_argument("--threads", type=int, default=4,
                       help="Core count to lock to (RPi 5 has 4)")
        p.add_argument("--mem-gb", type=float, default=4.0,
                       help="Simulated RAM limit (RPi 5 4GB)")
        p.add_argument("--conf", type=float, default=0.35)
        p.add_argument("--iou", type=float, default=0.45)
        p.add_argument("--rpi-factor", type=float, default=0.55,
                       help="Est. Pi5 FPS = raw laptop FPS * factor")

    sp = sub.add_parser("speed", help="FPS / latency benchmark on a video")
    _shared(sp)
    sp.add_argument("--models", nargs="+", required=True,
                    help="Model path(s): .pt, or exported NCNN/ONNX folders")
    sp.add_argument("--video", type=str, required=True)
    sp.add_argument("--imgsz", type=str, default="640,416,352",
                    help="Comma-separated input sizes to sweep")
    sp.add_argument("--frames", type=int, default=120)
    sp.set_defaults(func=run_speed)

    ap = sub.add_parser("accuracy", help="mAP on the val split")
    _shared(ap)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--data", type=str,
                    default="datasets/usable/yolo-human/data.yaml")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.set_defaults(func=run_accuracy)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()