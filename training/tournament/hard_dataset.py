"""
hard_dataset.py
===============
Hard validation set generator and curator for drone-footage annotation models.
Identifies and extracts difficult frames:
  - tiny / distant objects
  - high-density crowds of small targets
  - blur / motion artifacts
  - shadows / extreme lighting / poor contrast
"""

from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class FrameDifficulty:
    stem: str
    img_path: Path
    lbl_path: Path
    num_boxes: int
    num_tiny_boxes: int
    tiny_box_ratio: float
    min_box_area: float
    median_box_area: float
    blur_score: float        # Lower = more blurry (variance of Laplacian approximation)
    contrast_score: float    # Lower = flatter / shadows / poor contrast
    composite_hardness: float  # Higher = harder frame


def analyze_annotation_file(
    lbl_path: Path,
    tiny_area_threshold: float = 0.0015,
) -> Tuple[int, int, float, float, float]:
    """
    Analyzes a YOLO label file (.txt).
    Returns (num_boxes, num_tiny_boxes, tiny_box_ratio, min_area, median_area).
    Normalized area is w * h (in 0..1 range).
    """
    if not lbl_path.exists() or lbl_path.stat().st_size == 0:
        return 0, 0, 0.0, 1.0, 1.0

    areas: List[float] = []
    try:
        lines = lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # YOLO format: class_id cx cy w h
                w = float(parts[3])
                h = float(parts[4])
                area = w * h
                if area > 0:
                    areas.append(area)
    except Exception:
        return 0, 0, 0.0, 1.0, 1.0

    if not areas:
        return 0, 0, 0.0, 1.0, 1.0

    num_boxes = len(areas)
    tiny_boxes = sum(1 for a in areas if a <= tiny_area_threshold)
    tiny_ratio = tiny_boxes / num_boxes
    min_area = min(areas)
    median_area = float(np.median(areas))

    return num_boxes, tiny_boxes, tiny_ratio, min_area, median_area


def compute_image_quality_scores(img_path: Path) -> Tuple[float, float]:
    """
    Computes blur score and contrast score for an image.
    Uses PIL and numpy without requiring heavy C++ extensions.
    - blur_score: Variance of 2D Laplacian operator (lower = blurrier).
    - contrast_score: Standard deviation of luminance (lower = flat/shadows).
    """
    try:
        with Image.open(img_path) as im:
            # Resize thumbnail for fast processing (~256px max)
            im.thumbnail((256, 256), Image.Resampling.BILINEAR)
            gray = im.convert("L")
            arr = np.asarray(gray, dtype=np.float32)

        contrast = float(np.std(arr))

        # Approximate discrete Laplacian convolution for blur detection
        # Kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        if arr.shape[0] >= 3 and arr.shape[1] >= 3:
            lap = (
                arr[:-2, 1:-1] + arr[2:, 1:-1] +
                arr[1:-1, :-2] + arr[1:-1, 2:] -
                4.0 * arr[1:-1, 1:-1]
            )
            blur = float(np.var(lap))
        else:
            blur = 100.0

        return blur, contrast
    except Exception:
        return 100.0, 50.0


def score_frame_hardness(
    stem: str,
    img_path: Path,
    lbl_path: Path,
    tiny_area_threshold: float = 0.0015,
) -> FrameDifficulty:
    """Computes a multi-factor difficulty score for a validation frame."""
    num_boxes, num_tiny, tiny_ratio, min_area, median_area = analyze_annotation_file(
        lbl_path, tiny_area_threshold
    )

    blur_score, contrast_score = compute_image_quality_scores(img_path)

    # Frame Difficulty Formula:
    # 1. Tiny box presence (most important for small-object annotation)
    tiny_score = tiny_ratio * 40.0
    # 2. Box density / crowded distant scenes
    density_score = min(num_boxes / 25.0, 1.0) * 20.0
    # 3. Very small median area score
    min_area_score = (1.0 - min(median_area / 0.005, 1.0)) * 20.0
    # 4. Blur penalty (blurrier = harder)
    # Typical sharp images have blur_score > 300; blurry < 100
    blur_hardness = (1.0 - min(blur_score / 300.0, 1.0)) * 10.0
    # 5. Low contrast / shadows penalty
    # Typical contrast is 40-70; shadow/dark is < 25
    contrast_hardness = (1.0 - min(contrast_score / 60.0, 1.0)) * 10.0

    composite = tiny_score + density_score + min_area_score + blur_hardness + contrast_hardness

    return FrameDifficulty(
        stem=stem,
        img_path=img_path,
        lbl_path=lbl_path,
        num_boxes=num_boxes,
        num_tiny_boxes=num_tiny,
        tiny_box_ratio=tiny_ratio,
        min_box_area=min_area,
        median_box_area=median_area,
        blur_score=blur_score,
        contrast_score=contrast_score,
        composite_hardness=composite,
    )


def build_hard_validation_dataset(
    data_yaml_path: str | Path,
    out_dir: Optional[str | Path] = None,
    max_frames: int = 150,
    tiny_area_threshold: float = 0.0015,
    copy_files: bool = False,
) -> Path:
    """
    Scans the validation split from data_yaml, ranks all frames by hardness,
    and extracts the top `max_frames` most challenging frames into `val_hard`.
    Returns the path to the generated `hard_data.yaml`.
    """
    data_yaml = Path(data_yaml_path).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")

    # Parse data.yaml to locate val images/labels
    import yaml
    with open(data_yaml, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    dataset_root = Path(meta.get("path", data_yaml.parent))
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()

    val_img_rel = meta.get("val", "val/images")
    val_img_dir = (dataset_root / val_img_rel).resolve() if not Path(val_img_rel).is_absolute() else Path(val_img_rel)

    # Derive val label directory
    val_lbl_dir = val_img_dir.parent / "labels"
    if not val_lbl_dir.exists():
        val_lbl_dir = dataset_root / "val" / "labels"

    if not val_img_dir.exists() or not val_lbl_dir.exists():
        raise FileNotFoundError(f"Validation images or labels not found at {val_img_dir} / {val_lbl_dir}")

    # Destination directories
    if out_dir is None:
        target_hard_dir = dataset_root / "val_hard"
    else:
        target_hard_dir = Path(out_dir).resolve()

    hard_img_dir = target_hard_dir / "images"
    hard_lbl_dir = target_hard_dir / "labels"
    hard_img_dir.mkdir(parents=True, exist_ok=True)
    hard_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Scan all validation image-label pairs
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_files = [p for p in val_img_dir.iterdir() if p.suffix.lower() in supported_exts]

    if not img_files:
        raise RuntimeError(f"No validation images found in {val_img_dir}")

    print(f"[hard-val] Scanning {len(img_files)} validation frames for drone challenge attributes...")
    difficulties: List[FrameDifficulty] = []

    for img_path in img_files:
        lbl_path = val_lbl_dir / f"{img_path.stem}.txt"
        diff = score_frame_hardness(
            stem=img_path.stem,
            img_path=img_path,
            lbl_path=lbl_path,
            tiny_area_threshold=tiny_area_threshold,
        )
        # We only consider frames with at least one human box for the hard validation set
        if diff.num_boxes > 0:
            difficulties.append(diff)

    # Sort descending by composite hardness
    difficulties.sort(key=lambda d: d.composite_hardness, reverse=True)
    selected = difficulties[:max_frames]

    # Clean existing target directory contents
    for f in hard_img_dir.iterdir():
        if f.is_file() or f.is_symlink():
            f.unlink()
    for f in hard_lbl_dir.iterdir():
        if f.is_file() or f.is_symlink():
            f.unlink()

    # Populate hard validation set (symlink or copy)
    populated = 0
    for d in selected:
        dst_img = hard_img_dir / d.img_path.name
        dst_lbl = hard_lbl_dir / d.lbl_path.name

        if copy_files:
            shutil.copy2(d.img_path, dst_img)
            if d.lbl_path.exists():
                shutil.copy2(d.lbl_path, dst_lbl)
        else:
            try:
                # Windows & Linux symlink support
                dst_img.symlink_to(d.img_path.resolve())
                if d.lbl_path.exists():
                    dst_lbl.symlink_to(d.lbl_path.resolve())
            except OSError:
                shutil.copy2(d.img_path, dst_img)
                if d.lbl_path.exists():
                    shutil.copy2(d.lbl_path, dst_lbl)
        populated += 1

    # Create hard_data.yaml
    names_spec = meta.get("names", {0: "human"})
    hard_yaml_path = target_hard_dir / "hard_data.yaml"
    hard_yaml_content = (
        f"# HARD VALIDATION SET (Curated for tiny objects, blur, distant targets, shadows)\n"
        f"path: {target_hard_dir.resolve()}\n"
        f"train: images\n"  # dummy train path pointing to images to satisfy YOLO loader
        f"val: images\n"
        f"names:\n"
    )
    if isinstance(names_spec, dict):
        for k, v in names_spec.items():
            hard_yaml_content += f"  {k}: {v}\n"
    elif isinstance(names_spec, list):
        for idx, v in enumerate(names_spec):
            hard_yaml_content += f"  {idx}: {v}\n"

    hard_yaml_path.write_text(hard_yaml_content, encoding="utf-8")

    print(f"[hard-val] Curated {populated} difficult frames -> {target_hard_dir}")
    print(f"[hard-val] hard_data.yaml: {hard_yaml_path}")
    return hard_yaml_path
