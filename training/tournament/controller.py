"""
controller.py
=============
Sequential tournament controller and state machine for overnight model discovery.
Implements:
  - Strict sequential training on single GPU (RTX 4050)
  - Successive-halving candidate pruning (Phase 1 -> Phase 2 -> Phase 3)
  - Adaptive GPU time reallocation based on elapsed wall-clock time
  - Early stopping for plateaued / diverging candidates
  - Automatic OOM recovery (batch halving + cache clear)
  - Checkpoint preservation across phases
  - Resumable tournament state via state.json
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .config import (
    CandidateConfig,
    TournamentConfig,
    VRAMProfile,
    build_phase2_shortlist_candidates,
    build_phase3_finalist_candidates,
    get_default_phase1_candidates,
    get_smoke_test_candidates,
)
from .hard_dataset import build_hard_validation_dataset
from .metrics import CandidateMetrics, evaluate_candidate_checkpoint


class TournamentCallbacks:
    """Event hooks for TUI and CLI progress updates."""
    def on_tournament_start(self, config: TournamentConfig): pass
    def on_phase_start(self, phase: int, candidates: List[CandidateConfig]): pass
    def on_candidate_start(self, candidate: CandidateConfig, index: int, total: int): pass
    def on_candidate_progress(self, candidate_id: str, epoch: int, total_epochs: int, loss: float, map50: float): pass
    def on_candidate_complete(self, candidate: CandidateConfig, metrics: CandidateMetrics): pass
    def on_phase_complete(self, phase: int, ranked_metrics: List[CandidateMetrics]): pass
    def on_tournament_complete(self, leaderboard: List[CandidateMetrics], winner: CandidateMetrics): pass
    def on_log(self, message: str, level: str = "info"): pass


@dataclass
class TournamentResult:
    """Final packaged results of the tournament run."""
    leaderboard: List[CandidateMetrics]
    top_finalists: List[CandidateMetrics]
    recommended_winner: Optional[CandidateMetrics]
    total_elapsed_seconds: float
    checkpoints_dir: Path
    reports_dir: Path
    state_file: Path


class TournamentController:
    """
    Orchestrates the overnight drone annotation model tournament end to end.
    """

    def __init__(
        self,
        config: TournamentConfig,
        callbacks: Optional[TournamentCallbacks] = None,
        mock_mode: bool = False,
    ):
        self.config = config
        self.callbacks = callbacks or TournamentCallbacks()
        self.mock_mode = mock_mode

        self.root_dir = Path.cwd()
        self.output_dir = (self.root_dir / config.output_dir).resolve()
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.state_file = self.output_dir / "state.json"

        # Tournament State Variables
        self.start_time: float = 0.0
        self.elapsed_seconds: float = 0.0
        self.completed_candidates: Dict[str, CandidateMetrics] = {}
        self.best_weights_map: Dict[str, str] = {}
        self.phase_results: Dict[int, List[CandidateMetrics]] = {1: [], 2: [], 3: []}
        self.current_phase: int = 1
        self.active_candidate_id: Optional[str] = None
        self.aborted: bool = False
        self.hard_data_yaml: Optional[Path] = None

        # Empirical training speed tracker (seconds per epoch per 1000 images)
        self.seconds_per_epoch_history: List[float] = []

    # ----------------------------------------------------------------------- #
    # State Persistence & Resume
    # ----------------------------------------------------------------------- #

    def save_state(self):
        """Serializes current tournament progress to state.json."""
        state = {
            "current_phase": self.current_phase,
            "elapsed_seconds": self.elapsed_seconds,
            "start_time": self.start_time,
            "hard_data_yaml": str(self.hard_data_yaml) if self.hard_data_yaml else None,
            "best_weights_map": self.best_weights_map,
            "completed_candidates": {
                cid: m.to_dict() for cid, m in self.completed_candidates.items()
            },
            "phase_results": {
                str(p): [m.to_dict() for m in metrics_list]
                for p, metrics_list in self.phase_results.items()
            },
        }
        temp_file = self.state_file.with_suffix(".tmp")
        temp_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temp_file.replace(self.state_file)

    def load_state(self) -> bool:
        """Loads previous state if state.json exists. Returns True if resumed."""
        if not self.state_file.exists():
            return False

        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            self.current_phase = data.get("current_phase", 1)
            self.elapsed_seconds = data.get("elapsed_seconds", 0.0)
            self.start_time = data.get("start_time", time.time())
            if data.get("hard_data_yaml"):
                self.hard_data_yaml = Path(data["hard_data_yaml"])
            self.best_weights_map = data.get("best_weights_map", {})
            self.completed_candidates = {
                cid: CandidateMetrics.from_dict(m_dict)
                for cid, m_dict in data.get("completed_candidates", {}).items()
            }
            self.phase_results = {
                int(p): [CandidateMetrics.from_dict(m_dict) for m_dict in m_list]
                for p, m_list in data.get("phase_results", {}).items()
            }
            self.callbacks.on_log(
                f"Resumed tournament state from {self.state_file} ({len(self.completed_candidates)} completed runs)",
                level="info"
            )
            return True
        except Exception as e:
            self.callbacks.on_log(f"Warning loading state.json: {e}", level="warning")
            return False

    # ----------------------------------------------------------------------- #
    # Adaptive GPU Time Budget Reallocation
    # ----------------------------------------------------------------------- #

    @property
    def remaining_budget_seconds(self) -> float:
        """Calculates seconds left in the overnight window."""
        elapsed = time.time() - self.start_time if self.start_time > 0 else self.elapsed_seconds
        return max(0.0, self.config.total_budget_seconds - elapsed)

    def calculate_adaptive_epochs(
        self,
        candidate_count: int,
        target_imgsz: int,
        base_epochs: int,
        phase: int,
    ) -> int:
        """
        Dynamically calculates safe epoch allocation for upcoming candidates
        based on remaining GPU time budget and observed seconds per epoch.
        """
        if self.config.smoke_test:
            return 2

        rem_sec = self.remaining_budget_seconds
        if rem_sec <= 0:
            return max(5, base_epochs // 2)

        # Budget share for this phase: Phase 2 gets ~40% of remaining, Phase 3 gets ~60%
        phase_budget_ratio = 0.45 if phase == 2 else 0.85
        available_sec_for_phase = rem_sec * phase_budget_ratio
        sec_per_candidate = available_sec_for_phase / max(1, candidate_count)

        # Estimate seconds per epoch: default ~35-45s per epoch at 640px on RTX 4050 for VisDrone
        # Scales roughly with (imgsz / 640)^1.8
        res_scale = (target_imgsz / 640.0) ** 1.8
        est_sec_per_epoch = 40.0 * res_scale

        if self.seconds_per_epoch_history:
            avg_observed = sum(self.seconds_per_epoch_history) / len(self.seconds_per_epoch_history)
            est_sec_per_epoch = max(15.0, avg_observed * res_scale)

        computed_epochs = int(sec_per_candidate / est_sec_per_epoch)
        # Clamp between reasonable bounds
        min_allowed = 12 if phase == 2 else 20
        max_allowed = 45 if phase == 2 else 75
        clamped_epochs = max(min_allowed, min(max_allowed, computed_epochs))

        self.callbacks.on_log(
            f"Adaptive Budget [Phase {phase}]: {rem_sec/3600:.2f}h remaining -> "
            f"allocating {clamped_epochs} epochs/candidate ({candidate_count} candidates @ {target_imgsz}px)",
            level="info"
        )
        return clamped_epochs

    # ----------------------------------------------------------------------- #
    # Execution Engine
    # ----------------------------------------------------------------------- #

    def run(self) -> TournamentResult:
        """Executes the 3-phase successive-halving tournament sequentially."""
        if not self.load_state():
            self.start_time = time.time()

        self.callbacks.on_tournament_start(self.config)

        # 1. Ensure Hard Validation Set is prepared
        data_yaml_path = self.root_dir / self.config.data_yaml
        if self.config.use_hard_val and data_yaml_path.exists():
            try:
                self.hard_data_yaml = build_hard_validation_dataset(
                    data_yaml_path=data_yaml_path,
                    max_frames=self.config.hard_val_max_frames,
                    tiny_area_threshold=self.config.hard_val_area_threshold,
                )
            except Exception as e:
                self.callbacks.on_log(f"Hard validation set generation skipped: {e}", level="warning")
                self.hard_data_yaml = None

        # ------------------------------------------------------------------- #
        # PHASE 1 — SCREENING
        # ------------------------------------------------------------------- #
        if self.current_phase <= 1:
            self.callbacks.on_log("\n" + "=" * 64, level="info")
            self.callbacks.on_log("  🏆 PHASE 1: SCREENING (Wide & Cheap Search @ 640px)", level="info")
            self.callbacks.on_log("=" * 64, level="info")

            if self.config.smoke_test:
                p1_candidates = get_smoke_test_candidates(self.config.vram_profile)
            else:
                p1_candidates = get_default_phase1_candidates(self.config.vram_profile)[
                    :self.config.phase1_candidates_count
                ]

            self.callbacks.on_phase_start(1, p1_candidates)
            self._execute_candidate_batch(p1_candidates, phase=1)

            # Rank Phase 1 candidates by Drone Annotation Score
            p1_ranked = sorted(
                self.phase_results[1],
                key=lambda m: m.composite_drone_score,
                reverse=True
            )
            for idx, m in enumerate(p1_ranked, 1):
                m.rank = idx

            self.callbacks.on_phase_complete(1, p1_ranked)
            self.current_phase = 2
            self.save_state()

        # ------------------------------------------------------------------- #
        # PHASE 2 — SHORTLIST
        # ------------------------------------------------------------------- #
        if self.current_phase <= 2 and not self.aborted:
            self.callbacks.on_log("\n" + "=" * 64, level="info")
            self.callbacks.on_log("  🎯 PHASE 2: SHORTLIST (Top Performers @ 960px Warm-Start)", level="info")
            self.callbacks.on_log("=" * 64, level="info")

            p1_ranked = sorted(
                self.phase_results[1],
                key=lambda m: m.composite_drone_score,
                reverse=True
            )

            # Convert top ranked candidates back to CandidateConfig for shortlist generator
            p1_winner_configs = []
            for m in p1_ranked:
                # Find matching config or reconstruct
                cfg = CandidateConfig(
                    id=m.candidate_id,
                    name=m.name,
                    family=m.family,
                    scale=m.scale,
                    weights=m.weights_path or f"{m.family}{m.scale}.pt",
                    phase=1,
                    imgsz=m.imgsz,
                    epochs=m.epochs_completed,
                )
                p1_winner_configs.append(cfg)

            shortlist_count = self.config.phase2_shortlist_count
            adaptive_epochs = self.calculate_adaptive_epochs(
                candidate_count=shortlist_count,
                target_imgsz=960,
                base_epochs=25,
                phase=2,
            )

            p2_candidates = build_phase2_shortlist_candidates(
                ranked_phase1_candidates=p1_winner_configs,
                best_weights_map=self.best_weights_map,
                vram=self.config.vram_profile,
                shortlist_count=shortlist_count,
                additional_epochs=adaptive_epochs,
            )

            self.callbacks.on_phase_start(2, p2_candidates)
            self._execute_candidate_batch(p2_candidates, phase=2)

            # Rank Phase 2 candidates
            p2_ranked = sorted(
                self.phase_results[2],
                key=lambda m: m.composite_drone_score,
                reverse=True
            )
            for idx, m in enumerate(p2_ranked, 1):
                m.rank = idx

            self.callbacks.on_phase_complete(2, p2_ranked)
            self.current_phase = 3
            self.save_state()

        # ------------------------------------------------------------------- #
        # PHASE 3 — FINALISTS
        # ------------------------------------------------------------------- #
        if self.current_phase <= 3 and not self.aborted:
            self.callbacks.on_log("\n" + "=" * 64, level="info")
            self.callbacks.on_log("  👑 PHASE 3: FINALISTS (Deep Training Top 2-3 @ High-Res)", level="info")
            self.callbacks.on_log("=" * 64, level="info")

            p2_ranked = sorted(
                self.phase_results[2],
                key=lambda m: m.composite_drone_score,
                reverse=True
            )

            p2_winner_configs = [
                CandidateConfig(
                    id=m.candidate_id,
                    name=m.name,
                    family=m.family,
                    scale=m.scale,
                    weights=m.weights_path,
                    phase=2,
                    imgsz=m.imgsz,
                    epochs=m.epochs_completed,
                )
                for m in p2_ranked
            ]

            finalist_count = self.config.phase3_finalists_count
            adaptive_epochs_p3 = self.calculate_adaptive_epochs(
                candidate_count=finalist_count,
                target_imgsz=960,
                base_epochs=35,
                phase=3,
            )

            p3_candidates = build_phase3_finalist_candidates(
                ranked_phase2_candidates=p2_winner_configs,
                best_weights_map=self.best_weights_map,
                vram=self.config.vram_profile,
                finalist_count=finalist_count,
                target_epochs_addition=adaptive_epochs_p3,
            )

            self.callbacks.on_phase_start(3, p3_candidates)
            self._execute_candidate_batch(p3_candidates, phase=3)

            p3_ranked = sorted(
                self.phase_results[3],
                key=lambda m: m.composite_drone_score,
                reverse=True
            )
            for idx, m in enumerate(p3_ranked, 1):
                m.rank = idx

            self.callbacks.on_phase_complete(3, p3_ranked)
            self.current_phase = 4
            self.save_state()

        # ------------------------------------------------------------------- #
        # Package Tournament Results & Leaderboard
        # ------------------------------------------------------------------- #
        all_metrics = list(self.completed_candidates.values())
        overall_leaderboard = sorted(
            all_metrics,
            key=lambda m: m.composite_drone_score,
            reverse=True
        )
        for idx, m in enumerate(overall_leaderboard, 1):
            m.rank = idx

        winner = overall_leaderboard[0] if overall_leaderboard else None
        if winner:
            winner.status = "winner"

        # Tag top finalists
        top_finalists = [m for m in overall_leaderboard if m.phase == 3][:3]
        for f in top_finalists:
            if f != winner:
                f.status = "finalist"

        total_time = time.time() - self.start_time
        self.callbacks.on_tournament_complete(overall_leaderboard, winner)

        return TournamentResult(
            leaderboard=overall_leaderboard,
            top_finalists=top_finalists,
            recommended_winner=winner,
            total_elapsed_seconds=total_time,
            checkpoints_dir=self.checkpoints_dir,
            reports_dir=self.output_dir,
            state_file=self.state_file,
        )

    # ----------------------------------------------------------------------- #
    # Sequential Candidate Execution & Early Stopping
    # ----------------------------------------------------------------------- #

    def _execute_candidate_batch(self, candidates: List[CandidateConfig], phase: int):
        """Executes candidate configurations strictly sequentially on single GPU."""
        for idx, candidate in enumerate(candidates, 1):
            # Check if already completed (for resume support)
            if candidate.id in self.completed_candidates:
                existing_metric = self.completed_candidates[candidate.id]
                self.callbacks.on_log(
                    f"[Resume] Skipping already completed candidate: {candidate.id} "
                    f"(Score: {existing_metric.composite_drone_score:.1f})",
                    level="info"
                )
                if existing_metric not in self.phase_results[phase]:
                    self.phase_results[phase].append(existing_metric)
                continue

            # Check time budget
            if self.remaining_budget_seconds <= 120 and not self.config.smoke_test:
                self.callbacks.on_log(
                    f"Time budget depleted ({self.remaining_budget_seconds:.0f}s left). "
                    f"Wrapping up tournament.",
                    level="warning"
                )
                self.aborted = True
                break

            self.active_candidate_id = candidate.id
            self.callbacks.on_candidate_start(candidate, idx, len(candidates))

            metrics = self._train_and_evaluate_candidate(candidate)
            self.completed_candidates[candidate.id] = metrics
            self.phase_results[phase].append(metrics)

            if metrics.weights_path:
                self.best_weights_map[candidate.id] = metrics.weights_path

            self.callbacks.on_candidate_complete(candidate, metrics)
            self.save_state()

        self.active_candidate_id = None

    def _train_and_evaluate_candidate(self, candidate: CandidateConfig) -> CandidateMetrics:
        """
        Executes training for a single candidate with OOM recovery and early stopping.
        Preserves best checkpoint to runs/tournament/checkpoints/<candidate.id>/best.pt.
        """
        candidate_dir = self.output_dir / "runs" / candidate.id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        permanent_checkpoint_dir = self.checkpoints_dir / candidate.id
        permanent_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        preserved_best_pt = permanent_checkpoint_dir / "best.pt"

        t0 = time.time()

        if self.mock_mode:
            # Deterministic simulation for test verification
            time.sleep(0.05)
            # Simulate metrics based on model scale, imgsz, and phase
            scale_bonus = {"n": 0.0, "s": 0.05, "m": 0.08, "l": 0.10}.get(candidate.scale, 0.0)
            res_bonus = 0.06 if candidate.imgsz >= 960 else 0.0
            aug_bonus = 0.04 if candidate.augment_preset in ("drone_aerial", "small_object") else 0.0

            recall = min(0.95, 0.72 + scale_bonus + res_bonus + aug_bonus + (candidate.phase * 0.02))
            precision = min(0.93, 0.75 + scale_bonus + (candidate.phase * 0.02))
            map50 = min(0.94, 0.76 + scale_bonus + res_bonus + (candidate.phase * 0.03))
            map50_95 = min(0.75, 0.50 + scale_bonus + res_bonus + (candidate.phase * 0.02))
            small_rec = min(0.92, recall * (0.80 + 0.15 * (candidate.imgsz / 1280.0)))

            # Create dummy checkpoint file
            preserved_best_pt.write_text("mock_checkpoint_weights", encoding="utf-8")

            train_elapsed = time.time() - t0
            metrics = CandidateMetrics(
                candidate_id=candidate.id,
                name=candidate.name,
                phase=candidate.phase,
                family=candidate.family,
                scale=candidate.scale,
                imgsz=candidate.imgsz,
                batch=candidate.batch,
                recall=round(recall, 4),
                precision=round(precision, 4),
                mAP50=round(map50, 4),
                mAP50_95=round(map50_95, 4),
                small_object_recall=round(small_rec, 4),
                hard_set_recall=round(recall * 0.78, 4),
                inference_latency_ms=round(6.0 + (candidate.imgsz / 100.0) * (2 if candidate.scale == "m" else 1), 2),
                fps=round(1000.0 / (8.0 + candidate.imgsz / 100.0), 1),
                model_size_mb=6.2 if candidate.scale == "n" else 19.5,
                parameters_m=2.6 if candidate.scale == "n" else 9.4,
                training_time_s=round(train_elapsed, 1),
                epochs_completed=candidate.epochs,
                weights_path=str(preserved_best_pt),
            )
            from .metrics import compute_composite_drone_score
            metrics.composite_drone_score = compute_composite_drone_score(
                metrics,
                w_recall=self.config.w_recall,
                w_small_recall=self.config.w_small_recall,
                w_map50_95=self.config.w_map50_95,
                w_precision=self.config.w_precision,
                w_hard_set=self.config.w_hard_set,
            )
            return metrics

        # Live Ultralytics Training Execution
        try:
            from ultralytics import YOLO
            import torch

            # Determine weights path
            start_weights = candidate.weights
            if not Path(start_weights).exists():
                # Fallback to base weights in project root or ultralytics auto-download
                local_base = self.root_dir / Path(start_weights).name
                if local_base.exists():
                    start_weights = str(local_base)

            # Auto OOM Retry Loop (halves batch if CUDA OOM occurs)
            current_batch = candidate.batch
            success = False
            oom_attempts = 0

            while not success and oom_attempts < 3:
                try:
                    self.callbacks.on_log(
                        f"Initializing {candidate.id} ({candidate.family}{candidate.scale}) "
                        f"from {start_weights} @ {candidate.imgsz}px (batch {current_batch})...",
                        level="info"
                    )

                    model = YOLO(start_weights)
                    train_args = candidate.get_ultralytics_args(
                        data_yaml=str(self.root_dir / self.config.data_yaml),
                        project_dir=str(self.output_dir / "runs"),
                    )
                    train_args["batch"] = current_batch
                    if self.config.device is not None:
                        train_args["device"] = self.config.device
                    if self.config.fraction < 1.0:
                        train_args["fraction"] = self.config.fraction

                    # Run training
                    model.train(**train_args)
                    success = True

                except torch.cuda.OutOfMemoryError as oom_err:
                    oom_attempts += 1
                    torch.cuda.empty_cache()
                    current_batch = max(1, current_batch // 2)
                    self.callbacks.on_log(
                        f"⚠️ CUDA Out of Memory on RTX 4050! Halving batch to {current_batch} and retrying... ({oom_attempts}/3)",
                        level="warning"
                    )
                    if oom_attempts >= 3:
                        raise oom_err
                except Exception as e:
                    self.callbacks.on_log(f"Training error on candidate {candidate.id}: {e}", level="error")
                    break

            train_elapsed = time.time() - t0
            if candidate.epochs > 0:
                self.seconds_per_epoch_history.append(train_elapsed / candidate.epochs)

            # Locate best.pt produced by Ultralytics run
            expected_best = self.output_dir / "runs" / candidate.id / "weights" / "best.pt"
            if not expected_best.exists():
                # Check for last.pt
                expected_best = self.output_dir / "runs" / candidate.id / "weights" / "last.pt"

            if expected_best.exists():
                # Preserve best checkpoint permanently in tournament checkpoints
                shutil.copy2(expected_best, preserved_best_pt)
                weights_to_eval = preserved_best_pt
            else:
                self.callbacks.on_log(f"No checkpoint produced for {candidate.id}, using base weights", level="warning")
                weights_to_eval = start_weights

            # Standardized Evaluation
            metrics = evaluate_candidate_checkpoint(
                weights_path=weights_to_eval,
                data_yaml=self.root_dir / self.config.data_yaml,
                hard_data_yaml=self.hard_data_yaml,
                imgsz=candidate.imgsz,
                device=self.config.device,
                candidate_id=candidate.id,
                phase=candidate.phase,
                name=candidate.name,
                family=candidate.family,
                scale=candidate.scale,
                batch=current_batch,
                training_time_s=round(train_elapsed, 1),
                w_recall=self.config.w_recall,
                w_small_recall=self.config.w_small_recall,
                w_map50_95=self.config.w_map50_95,
                w_precision=self.config.w_precision,
                w_hard_set=self.config.w_hard_set,
            )

            # Clean large intermediate optimizer states to save disk space on host
            # (keeps best.pt and results.csv)
            last_pt = self.output_dir / "runs" / candidate.id / "weights" / "last.pt"
            if last_pt.exists() and last_pt != expected_best:
                try:
                    last_pt.unlink()
                except Exception:
                    pass

            return metrics

        except Exception as e:
            self.callbacks.on_log(f"Fatal error during candidate {candidate.id}: {e}", level="error")
            train_elapsed = time.time() - t0
            return CandidateMetrics(
                candidate_id=candidate.id,
                name=candidate.name,
                phase=candidate.phase,
                family=candidate.family,
                scale=candidate.scale,
                imgsz=candidate.imgsz,
                batch=candidate.batch,
                training_time_s=train_elapsed,
                notes=f"Error: {e}",
            )
