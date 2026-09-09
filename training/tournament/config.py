"""
config.py
=========
Configuration dataclasses, VRAM profiles for RTX 4050 (6 GB), candidate spaces,
and tournament presets for the Overnight Drone-Footage Model Tournament.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class TournamentPreset(str, Enum):
    OVERNIGHT_STANDARD = "overnight_standard"  # ~6.5 hours (default)
    OVERNIGHT_EXTENDED = "overnight_extended"  # ~8.0 hours
    FAST_SCREENING = "fast_screening"          # ~2.5 hours
    SMOKE_TEST = "smoke_test"                  # ~10-15 minutes (pipeline verification)


@dataclass
class VRAMProfile:
    """
    VRAM management profile tailored for laptop NVIDIA RTX 4050 (6 GB VRAM).
    Provides safe maximum batch sizes for training without out-of-memory (OOM) crashes.
    """
    total_vram_gb: float = 6.0
    safe_headroom_gb: float = 0.8  # Reserve for OS display, PyTorch context, and peak memory

    # Safe batch size lookups: (model_scale, imgsz) -> batch_size
    _BATCH_TABLE: Dict[str, Dict[int, int]] = field(default_factory=lambda: {
        # Nano models (~2.6 - 3.2M params)
        "n": {
            320: 32,
            512: 24,
            640: 16,
            960: 10,
            1280: 4,
        },
        # Small models (~9.4 - 11.2M params)
        "s": {
            320: 24,
            512: 16,
            640: 12,
            960: 6,
            1280: 3,
        },
        # Medium models (~20 - 26M params)
        "m": {
            320: 16,
            512: 10,
            640: 8,
            960: 4,
            1280: 2,
        },
        # Large models (~43 - 54M params) - used sparingly for finalists
        "l": {
            320: 8,
            512: 6,
            640: 4,
            960: 2,
            1280: 1,
        },
    })

    def get_safe_batch(self, model_scale: str, imgsz: int) -> int:
        """Returns safe batch size for the given scale (n/s/m/l) and image resolution."""
        scale = model_scale.lower()[-1] if model_scale else "n"
        if scale not in self._BATCH_TABLE:
            scale = "n"
        table = self._BATCH_TABLE[scale]
        # Find nearest resolution <= requested imgsz
        res_keys = sorted(table.keys())
        chosen_res = res_keys[0]
        for r in res_keys:
            if imgsz >= r:
                chosen_res = r
            else:
                break
        return table[chosen_res]


@dataclass
class CandidateConfig:
    """
    Configuration specification for a single candidate model run in the tournament.
    """
    id: str                                # Unique identifier, e.g. "C01-yolo11n-base"
    name: str                              # Descriptive display name
    family: str                            # "yolo11", "yolo26", "yolo8"
    scale: str                             # "n", "s", "m", "l"
    weights: str                           # Initial weights / checkpoint path
    phase: int                             # 1 = Screening, 2 = Shortlist, 3 = Finalists
    imgsz: int = 640                       # Input resolution
    batch: int = 16                        # Batch size (VRAM tuned)
    epochs: int = 20                       # Target epochs for this phase
    freeze: int = 10                       # Backbone frozen layers (10 = frozen, 0 = full tune)
    single_cls: bool = True                # Single human class
    cos_lr: bool = False                   # Cosine learning rate scheduler
    lr0: float = 0.01                      # Initial learning rate
    lrf: float = 0.01                      # Final learning rate fraction
    augment_preset: str = "standard"       # "standard", "drone_aerial", "small_object"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    parent_candidate_id: Optional[str] = None  # Previous phase candidate if warm-started
    description: str = ""

    def get_ultralytics_args(self, data_yaml: str, project_dir: str) -> Dict[str, Any]:
        """Translates candidate config into Ultralytics YOLO training arguments."""
        args: Dict[str, Any] = {
            "data": str(data_yaml),
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "freeze": self.freeze,
            "single_cls": self.single_cls,
            "cos_lr": self.cos_lr,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "project": str(project_dir),
            "name": self.id,
            "exist_ok": True,
            "verbose": False,
            "save": True,
            "plots": True,
            "val": True,
        }

        # Apply augmentation preset overrides
        if self.augment_preset == "drone_aerial":
            # Augmentations tuned for aerial drone footage: perspective, rotational view, scales
            args.update({
                "mosaic": 1.0,
                "mixup": 0.15,
                "degrees": 10.0,       # Drone camera tilt / rotation
                "translate": 0.15,     # High flight drift
                "scale": 0.5,          # Large scale variation (high vs low altitude)
                "shear": 2.0,          # Oblique perspective
                "perspective": 0.0005, # Drone oblique ground plane
                "flipud": 0.5,         # Overhead drone viewpoint is orientation-agnostic
                "fliplr": 0.5,
                "close_mosaic": 8,
            })
        elif self.augment_preset == "small_object":
            # Augmentations and losses tuned for tiny distant targets
            args.update({
                "mosaic": 1.0,
                "mixup": 0.10,
                "scale": 0.7,          # Heavy multi-scale simulation
                "box": 8.5,            # Increased box loss gain for tight localization
                "cls": 0.6,
                "dfl": 1.8,            # Distribution focal loss gain for sub-pixel edges
                "close_mosaic": 10,    # Turn off mosaic earlier to refine small bounding boxes
                "flipud": 0.5,
                "fliplr": 0.5,
            })
        else:
            # Standard baseline
            args.update({
                "mosaic": 1.0,
                "mixup": 0.0,
                "close_mosaic": 5,
                "fliplr": 0.5,
            })

        # Apply any custom hyperparameter overrides
        if self.hyperparameters:
            args.update(self.hyperparameters)

        return args

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CandidateConfig:
        return cls(**data)


@dataclass
class TournamentConfig:
    """
    Top-level configuration for the entire overnight tournament execution.
    """
    preset: TournamentPreset = TournamentPreset.OVERNIGHT_STANDARD
    total_budget_hours: float = 6.5
    data_yaml: str = "datasets/usable/yolo-human/data.yaml"
    output_dir: str = "runs/tournament"

    # Hardware & VRAM constraints
    vram_profile: VRAMProfile = field(default_factory=VRAMProfile)
    device: Optional[str] = None  # None = auto (GPU 0 if CUDA available)
    workers: int = 8

    # Drone Annotation Priority Weights
    # Offline annotation priority: Recall > Small Object Recall > mAP50-95 > Precision > Hard Set
    w_recall: float = 0.35          # Missed objects can never be curated
    w_small_recall: float = 0.25    # Small / distant objects are the hardest failure mode
    w_map50_95: float = 0.20        # General box localization accuracy
    w_precision: float = 0.15       # Bounding box quality / minimal false alarms
    w_hard_set: float = 0.05        # Bonus for robustness on hard validation frames

    # Phasing & Selection
    phase1_candidates_count: int = 8
    phase2_shortlist_count: int = 4
    phase3_finalists_count: int = 2

    # Adaptive Early Stopping
    min_epochs_before_stop: int = 6
    divergence_loss_threshold: float = 18.0
    patience_epochs: int = 8

    # Hard Validation Subset Configuration
    use_hard_val: bool = True
    hard_val_max_frames: int = 150
    hard_val_area_threshold: float = 0.0015  # Normalized box area threshold for small objects

    # System & Execution Flags
    prevent_sleep: bool = True      # Prevent Windows system sleep while running
    smoke_test: bool = False        # Tiny verification mode
    fraction: float = 1.0           # Dataset fraction (e.g. 0.05 for smoke test)

    @property
    def total_budget_seconds(self) -> float:
        return self.total_budget_hours * 3600.0


# --------------------------------------------------------------------------- #
# Default Candidate Spaces
# --------------------------------------------------------------------------- #

def get_default_phase1_candidates(vram: VRAMProfile) -> List[CandidateConfig]:
    """
    Returns the Phase 1 (Screening) candidate space:
    Diverse model families, sizes, augmentations, and small-object tunings at 640px.
    """
    candidates = [
        # 1. YOLO11n Baseline
        CandidateConfig(
            id="P1-01-yolo11n-base",
            name="YOLO11n Baseline (640px)",
            family="yolo11",
            scale="n",
            weights="yolo11n.pt",
            phase=1,
            imgsz=640,
            batch=vram.get_safe_batch("n", 640),
            epochs=18,
            freeze=10,
            augment_preset="standard",
            description="Standard YOLO11n fine-tune with frozen backbone as reference baseline.",
        ),
        # 2. YOLO11n Drone Aerial Augmentation
        CandidateConfig(
            id="P1-02-yolo11n-aerial",
            name="YOLO11n Drone Aerial (640px)",
            family="yolo11",
            scale="n",
            weights="yolo11n.pt",
            phase=1,
            imgsz=640,
            batch=vram.get_safe_batch("n", 640),
            epochs=18,
            freeze=10,
            augment_preset="drone_aerial",
            description="YOLO11n with rotational invariance, perspective tilt, and aerial scale jitter.",
        ),
        # 3. YOLO11n Small-Object Tuning
        CandidateConfig(
            id="P1-03-yolo11n-smallobj",
            name="YOLO11n Small-Object (640px)",
            family="yolo11",
            scale="n",
            weights="yolo11n.pt",
            phase=1,
            imgsz=640,
            batch=vram.get_safe_batch("n", 640),
            epochs=18,
            freeze=10,
            augment_preset="small_object",
            description="YOLO11n tuned for distant human detection with boosted box/DFL loss.",
        ),
        # 4. YOLO11s Baseline (Higher capacity)
        CandidateConfig(
            id="P1-04-yolo11s-base",
            name="YOLO11s Baseline (640px)",
            family="yolo11",
            scale="s",
            weights="yolo11s.pt",
            phase=1,
            imgsz=640,
            batch=vram.get_safe_batch("s", 640),
            epochs=18,
            freeze=10,
            augment_preset="standard",
            description="YOLO11s (9.4M params) offering greater feature capacity for small targets.",
        ),
        # 5. YOLO11s Drone Aerial Augmentation
        CandidateConfig(
            id="P1-05-yolo11s-aerial",
            name="YOLO11s Drone Aerial (640px)",
            family="yolo11",
            scale="s",
            weights="yolo11s.pt",
            phase=1,
            imgsz=640,
            batch=vram.get_safe_batch("s", 640),
            epochs=18,
            freeze=10,
            augment_preset="drone_aerial",
            description="YOLO11s trained with heavy perspective and rotation invariance for drones.",
        ),
        # 6. YOLO11s Small-Object Tuning
        CandidateConfig(
            id="P1-06-yolo11s-smallobj",
            name="YOLO11s Small-Object (640px)",
            family="yolo11",
            scale="s",
            weights="yolo11s.pt",
            phase=1,
            imgsz=640,
            batch=vram.get_safe_batch("s", 640),
            epochs=18,
            freeze=10,
            augment_preset="small_object",
            description="YOLO11s small-object specialist with high box weight and early mosaic close.",
        ),
        # 7. YOLO11m Aerial Specialist (Medium capacity)
        CandidateConfig(
            id="P1-07-yolo11m-aerial",
            name="YOLO11m Aerial (640px)",
            family="yolo11",
            scale="m",
            weights="yolo11m.pt",
            phase=1,
            imgsz=640,
            batch=vram.get_safe_batch("m", 640),
            epochs=15,
            freeze=10,
            augment_preset="drone_aerial",
            description="Medium model (20M params) for deep feature representations of tiny humans.",
        ),
        # 8. YOLO26n / End-to-End Specialist
        CandidateConfig(
            id="P1-08-yolo26n-smallobj",
            name="YOLO26n End-to-End Small-Obj (640px)",
            family="yolo26",
            scale="n",
            weights="yolo26n.pt",
            phase=1,
            imgsz=640,
            batch=vram.get_safe_batch("n", 640),
            epochs=18,
            freeze=10,
            augment_preset="small_object",
            description="Dual-assignment NMS-free architecture with small-object focus.",
        ),
    ]
    return candidates


def get_smoke_test_candidates(vram: VRAMProfile) -> List[CandidateConfig]:
    """
    Returns a minimal 2-candidate, 2-epoch candidate list for end-to-end smoke testing.
    """
    return [
        CandidateConfig(
            id="SMOKE-01-yolo11n",
            name="Smoke Test YOLO11n (320px)",
            family="yolo11",
            scale="n",
            weights="yolo11n.pt",
            phase=1,
            imgsz=320,
            batch=8,
            epochs=2,
            freeze=10,
            augment_preset="standard",
            description="Fast smoke test candidate for pipeline verification.",
        ),
        CandidateConfig(
            id="SMOKE-02-yolo11s",
            name="Smoke Test YOLO11s (320px)",
            family="yolo11",
            scale="s",
            weights="yolo11s.pt",
            phase=1,
            imgsz=320,
            batch=4,
            epochs=2,
            freeze=10,
            augment_preset="drone_aerial",
            description="Fast smoke test candidate with aerial augmentations.",
        ),
    ]


def build_phase2_shortlist_candidates(
    ranked_phase1_candidates: List[CandidateConfig],
    best_weights_map: Dict[str, str],
    vram: VRAMProfile,
    shortlist_count: int = 4,
    additional_epochs: int = 25,
) -> List[CandidateConfig]:
    """
    Creates Phase 2 (Shortlist) candidates from top Phase 1 winners:
    - Warm-starts from their Phase 1 best weights
    - Explores higher resolution (960px) where promising, or continues 640px
    - Adjusts batch size safely for VRAM
    - Unfreezes backbone (freeze=0) or continues with cosine LR
    """
    phase2_candidates: List[CandidateConfig] = []
    top_candidates = ranked_phase1_candidates[:shortlist_count]

    for idx, c in enumerate(top_candidates):
        best_pt = best_weights_map.get(c.id, c.weights)
        p2_id = f"P2-{idx+1:02d}-{c.family}{c.scale}-{c.augment_preset}"

        # Give top 2 candidates higher resolution (960px) for small-object exploration!
        use_960 = idx < 2
        imgsz = 960 if use_960 else 640
        safe_batch = vram.get_safe_batch(c.scale, imgsz)

        p2_candidate = CandidateConfig(
            id=p2_id,
            name=f"{c.name.split(' (')[0]} Shortlist ({imgsz}px)",
            family=c.family,
            scale=c.scale,
            weights=best_pt,
            phase=2,
            imgsz=imgsz,
            batch=safe_batch,
            epochs=c.epochs + additional_epochs,
            freeze=0,  # Unfreeze backbone for full fine-tuning
            cos_lr=True,
            lr0=0.005,  # Lower LR for fine-tuning
            augment_preset=c.augment_preset,
            hyperparameters=copy.deepcopy(c.hyperparameters),
            parent_candidate_id=c.id,
            description=f"Phase 2 Shortlist warm-started from {c.id} best weights @ {imgsz}px.",
        )
        phase2_candidates.append(p2_candidate)

    return phase2_candidates


def build_phase3_finalist_candidates(
    ranked_phase2_candidates: List[CandidateConfig],
    best_weights_map: Dict[str, str],
    vram: VRAMProfile,
    finalist_count: int = 2,
    target_epochs_addition: int = 35,
) -> List[CandidateConfig]:
    """
    Creates Phase 3 (Finalists) candidates from top Phase 2 winners:
    - Deep training on the strongest configurations
    - Explores high resolution (960px or 1280px for small-object candidates)
    - Full cosine annealing
    """
    phase3_candidates: List[CandidateConfig] = []
    top_candidates = ranked_phase2_candidates[:finalist_count]

    for idx, c in enumerate(top_candidates):
        best_pt = best_weights_map.get(c.id, c.weights)
        p3_id = f"P3-{idx+1:02d}-FINALIST-{c.family}{c.scale}"

        # Top finalist gets 1280px if small-object specialist or nano/small, else 960px
        is_small_obj = "small" in c.augment_preset or "smallobj" in c.id
        imgsz = 1280 if (is_small_obj and c.scale in ("n", "s") and idx == 0) else 960
        safe_batch = vram.get_safe_batch(c.scale, imgsz)

        p3_candidate = CandidateConfig(
            id=p3_id,
            name=f"🏆 Finalist: {c.name.split(' (')[0]} ({imgsz}px)",
            family=c.family,
            scale=c.scale,
            weights=best_pt,
            phase=3,
            imgsz=imgsz,
            batch=safe_batch,
            epochs=c.epochs + target_epochs_addition,
            freeze=0,
            cos_lr=True,
            lr0=0.002,  # Fine refinement LR
            lrf=0.005,
            augment_preset=c.augment_preset,
            hyperparameters=copy.deepcopy(c.hyperparameters),
            parent_candidate_id=c.id,
            description=f"Phase 3 Tournament Finalist warm-started from {c.id} @ {imgsz}px.",
        )
        phase3_candidates.append(p3_candidate)

    return phase3_candidates


def get_preset_config(preset: TournamentPreset) -> TournamentConfig:
    """Returns a pre-configured TournamentConfig for the specified preset."""
    if preset == TournamentPreset.SMOKE_TEST:
        return TournamentConfig(
            preset=TournamentPreset.SMOKE_TEST,
            total_budget_hours=0.25,  # 15 minutes
            phase1_candidates_count=2,
            phase2_shortlist_count=1,
            phase3_finalists_count=1,
            fraction=0.05,
            smoke_test=True,
            min_epochs_before_stop=2,
            patience_epochs=2,
        )
    elif preset == TournamentPreset.FAST_SCREENING:
        return TournamentConfig(
            preset=TournamentPreset.FAST_SCREENING,
            total_budget_hours=2.5,
            phase1_candidates_count=6,
            phase2_shortlist_count=3,
            phase3_finalists_count=2,
            fraction=1.0,
            smoke_test=False,
        )
    elif preset == TournamentPreset.OVERNIGHT_EXTENDED:
        return TournamentConfig(
            preset=TournamentPreset.OVERNIGHT_EXTENDED,
            total_budget_hours=8.0,
            phase1_candidates_count=8,
            phase2_shortlist_count=4,
            phase3_finalists_count=3,
            fraction=1.0,
            smoke_test=False,
        )
    else:  # OVERNIGHT_STANDARD
        return TournamentConfig(
            preset=TournamentPreset.OVERNIGHT_STANDARD,
            total_budget_hours=6.5,
            phase1_candidates_count=8,
            phase2_shortlist_count=4,
            phase3_finalists_count=2,
            fraction=1.0,
            smoke_test=False,
        )
