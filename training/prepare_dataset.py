#!/usr/bin/env python3
"""
prepare_dataset.py
==================
Convert the raw VisDrone2019-DET annotations + images into a YOLO-format
dataset ready for `ultralytics` / YOLO11 training.

VisDrone annotation format (per line):
    x, y, width, height, score, category, truncation, occlusion
    where (x, y) is the TOP-LEFT corner in PIXELS (absolute, not normalized).

We consolidate the two human classes into a single class:
    VisDrone category 1 (pedestrian)  -> 0  (human)
    VisDrone category 2 (people)      -> 0  (human)
Everything else (cars, vans, trucks, bicycles, ignored-regions, ...) is
discarded so the model learns to detect ONLY humans from a drone view.

Output layout (YOLO format, labels are normalized center-x, center-y, w, h):
    <output_dir>/train/images/*.jpg
    <output_dir>/train/labels/*.txt
    <output_dir>/val/images/*.jpg
    <output_dir>/val/labels/*.txt
    <output_dir>/data.yaml

Usage:
    python training/prepare_dataset.py
    python training/prepare_dataset.py --train ../datasets/usable/VisDrone2019-DET-train \
                                       --val   ../datasets/usable/VisDrone2019-DET-val \
                                       --out   ../datasets/yolo-human
"""

import argparse
import shutil
from pathlib import Path

from PIL import Image

# VisDrone category ids that we keep and map onto a single "human" class.
HUMAN_CATEGORIES = {1: 0, 2: 0}  # pedestrian, people -> class 0 (human)

SKIP_CATEGORY = 0  # "ignored regions" are never treated as humans


def parse_args():
    p = argparse.ArgumentParser(description="Convert VisDrone2019-DET to YOLO format (humans only)")
    p.add_argument("--train", type=str,
                   default="datasets/usable/VisDrone2019-DET-train",
                   help="Path to VisDrone train split (folder with images/ and annotations/)")
    p.add_argument("--val", type=str,
                   default="datasets/usable/VisDrone2019-DET-val",
                   help="Path to VisDrone val split")
    p.add_argument("--out", type=str,
                   default="datasets/usable/yolo-human",
                   help="Output folder for the YOLO-format dataset")
    p.add_argument("--copy", action="store_true",
                   help="Copy images instead of symlinking them (default: symlink to save disk)")
    p.add_argument("--only-with-humans", action="store_true",
                   help="Only keep images/anns that contain at least one human box "
                        "(default keeps every frame, including negatives, so the model "
                        "learns not to fire on cars/buildings)")
    return p.parse_args()


def load_human_boxes(ann_path: Path):
    """
    Read a VisDrone .txt annotation and return a list of
    (x, y, w, h, class_id) in PIXEL coords for the human class.
    Boxes that are degenerate (w or h <= 0) or belong to ignored /
    non-human categories are skipped.
    """
    boxes = []
    for raw in ann_path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        vals = line.split(",")
        if len(vals) < 6:
            continue
        try:
            x, y, w, h, _score, cat = (float(vals[0]), float(vals[1]),
                                       float(vals[2]), float(vals[3]),
                                       float(vals[4]), int(vals[5]))
        except ValueError:
            continue

        if cat == SKIP_CATEGORY or cat not in HUMAN_CATEGORIES:
            continue
        if w <= 0 or h <= 0:
            continue

        boxes.append((x, y, w, h, HUMAN_CATEGORIES[cat]))

    return boxes


def convert_split(src: Path, dst: Path, copy: bool, only_with_humans: bool):
    """
    Convert one split (train/val). Returns the number of converted images.
    """
    ann_dir = src / "annotations"
    img_dir = src / "images"

    dst_img = dst / "images"
    dst_lbl = dst / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(ann_dir.glob("*.txt"))
    converted = 0
    skipped_no_humans = 0

    for ann in ann_files:
        img_path = img_dir / (ann.stem + ".jpg")
        if not img_path.exists():
            continue

        boxes = load_human_boxes(ann)
        if only_with_humans and not boxes:
            skipped_no_humans += 1
            continue

        # Symlink or copy the image
        link = dst_img / img_path.name
        if copy:
            shutil.copy2(img_path, link)
        elif not link.exists() and not link.is_symlink():
            try:
                link.symlink_to(img_path.resolve())
            except OSError:
                shutil.copy2(img_path, link)

        # Write YOLO label file (normalized center-xywh, each in 0..1).
        # VisDrone images can vary in size, so read the real dimensions.
        with Image.open(img_path) as im:
            iw, ih = im.size
        lbl_path = dst_lbl / (ann.stem + ".txt")
        lines = []
        for (x, y, w, h, cls) in boxes:
            cx = (x + w / 2.0) / iw
            cy = (y + h / 2.0) / ih
            wn = w / iw
            hn = h / ih
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")
        lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""))

        converted += 1

    print(f"[{src.name}] converted {converted} images"
          f"{f' (skipped {skipped_no_humans} without humans)' if only_with_humans else ''}")
    return converted


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    train_src = (root / args.train) if not Path(args.train).is_absolute() else Path(args.train)
    val_src = (root / args.val) if not Path(args.val).is_absolute() else Path(args.val)
    out = (root / args.out) if not Path(args.out).is_absolute() else Path(args.out)

    n_train = convert_split(train_src, out / "train", args.copy, args.only_with_humans)
    n_val = convert_split(val_src, out / "val", args.copy, args.only_with_humans)

    # Absolute paths so training works regardless of the working directory.
    # `train: train/images` -> <path>/train/images (labels live in <path>/train/labels).
    data_yaml = (
        "# YOLO dataset config: humans (from VisDrone2019-DET drone view)\n"
        f"path: {out.resolve()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "names:\n"
        "  0: human\n"
    )
    yaml_path = out / "data.yaml"
    yaml_path.write_text(data_yaml)

    print("\nDone.")
    print(f"  train images : {n_train}")
    print(f"  val images   : {n_val}")
    print(f"  data.yaml    : {yaml_path}")
    print("\nNext step:  python training/train.py")


if __name__ == "__main__":
    main()

