#!/usr/bin/env python3
"""Helpers used by 5_package_results.ps1.

Subcommands
-----------
  video    Synthesize a sample drone clip from val frames (for the FPS benchmark).
  preview  Save an annotated prediction grid over val frames.
"""

import argparse
import glob
import os


def cmd_video(a):
    import cv2

    frames = sorted(glob.glob(os.path.join(a.val_dir, "images", "*.jpg")))[: a.frames]
    if not frames:
        raise SystemExit(f"no jpg frames found under {a.val_dir}")
    w, h = 1280, 720
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    n = 0
    for f in frames:
        im = cv2.imread(f)
        if im is None:
            continue
        vw.write(cv2.resize(im, (w, h)))
        n += 1
    vw.release()
    print(f"[video] wrote {n} frames -> {a.out}")


def cmd_preview(a):
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ultralytics import YOLO

    samples = sorted(glob.glob(os.path.join(a.val_dir, "images", "*.jpg")))[:6]
    if not samples:
        raise SystemExit(f"no jpg frames found under {a.val_dir}")
    model = YOLO(a.model)
    results = model.predict(samples, imgsz=a.imgsz, conf=0.35, iou=0.45, verbose=False)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, r in zip(axes.flat, results):
        ax.imshow(cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB))
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    plt.close(fig)
    print(f"[preview] wrote {len(results)} predictions -> {a.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("video", help="synthesize sample video from val frames")
    v.add_argument("--val-dir", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--frames", type=int, default=120)
    v.set_defaults(func=cmd_video)

    pr = sub.add_parser("preview", help="annotated prediction grid")
    pr.add_argument("--model", required=True)
    pr.add_argument("--val-dir", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--imgsz", type=int, default=416)
    pr.set_defaults(func=cmd_preview)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
