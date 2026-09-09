"""
test_tournament.py
==================
Unit and integration tests for the Overnight Drone-Footage Model Tournament.
Validates:
  - VRAM profile batch safety for RTX 4050 (6 GB)
  - Candidate spaces, augmentations, and phase transitions
  - Composite Drone Annotation Quality Score calculations
  - Hard validation set analysis logic
  - Controller state persistence, dynamic budgeting, and mock execution
  - Report and leaderboard generation
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from training.tournament.config import (
    CandidateConfig,
    TournamentConfig,
    TournamentPreset,
    VRAMProfile,
    build_phase2_shortlist_candidates,
    build_phase3_finalist_candidates,
    get_default_phase1_candidates,
    get_preset_config,
    get_smoke_test_candidates,
)
from training.tournament.controller import TournamentController
from training.tournament.hard_dataset import analyze_annotation_file, score_frame_hardness
from training.tournament.metrics import (
    CandidateMetrics,
    compute_composite_drone_score,
)
from training.tournament.platform_utils import (
    WindowsSleepInhibitor,
    enable_windows_virtual_terminal,
    get_project_python_exe,
)
from training.tournament.reporter import (
    format_time_duration,
    generate_leaderboard_markdown,
    save_tournament_reports,
)


class TestVRAMProfile(unittest.TestCase):
    def setUp(self):
        self.vram = VRAMProfile(total_vram_gb=6.0)

    def test_safe_batch_sizes(self):
        # Nano models
        self.assertEqual(self.vram.get_safe_batch("n", 640), 16)
        self.assertEqual(self.vram.get_safe_batch("n", 960), 10)
        self.assertEqual(self.vram.get_safe_batch("n", 1280), 4)

        # Small models
        self.assertEqual(self.vram.get_safe_batch("s", 640), 12)
        self.assertEqual(self.vram.get_safe_batch("s", 960), 6)
        self.assertEqual(self.vram.get_safe_batch("s", 1280), 3)

        # Medium models
        self.assertEqual(self.vram.get_safe_batch("m", 640), 8)
        self.assertEqual(self.vram.get_safe_batch("m", 960), 4)
        self.assertEqual(self.vram.get_safe_batch("m", 1280), 2)


class TestCandidateConfig(unittest.TestCase):
    def test_candidate_ultralytics_args(self):
        c = CandidateConfig(
            id="test-c01",
            name="Test Candidate",
            family="yolo11",
            scale="s",
            weights="yolo11s.pt",
            phase=1,
            imgsz=640,
            batch=12,
            epochs=20,
            augment_preset="drone_aerial",
        )
        args = c.get_ultralytics_args(data_yaml="data.yaml", project_dir="runs")
        self.assertEqual(args["imgsz"], 640)
        self.assertEqual(args["batch"], 12)
        self.assertEqual(args["epochs"], 20)
        self.assertEqual(args["single_cls"], True)
        self.assertIn("perspective", args)
        self.assertIn("degrees", args)
        self.assertEqual(args["flipud"], 0.5)

    def test_small_object_preset(self):
        c = CandidateConfig(
            id="test-smallobj",
            name="Test Small Obj",
            family="yolo11",
            scale="n",
            weights="yolo11n.pt",
            phase=1,
            augment_preset="small_object",
        )
        args = c.get_ultralytics_args(data_yaml="data.yaml", project_dir="runs")
        self.assertEqual(args["box"], 8.5)
        self.assertEqual(args["dfl"], 1.8)
        self.assertEqual(args["close_mosaic"], 10)


class TestPhaseProgression(unittest.TestCase):
    def setUp(self):
        self.vram = VRAMProfile()

    def test_phase1_candidates_count(self):
        p1 = get_default_phase1_candidates(self.vram)
        self.assertEqual(len(p1), 8)
        scales = {c.scale for c in p1}
        self.assertIn("n", scales)
        self.assertIn("s", scales)
        self.assertIn("m", scales)

    def test_phase2_shortlist_builder(self):
        p1 = get_default_phase1_candidates(self.vram)
        best_map = {c.id: f"runs/detect/{c.id}/weights/best.pt" for c in p1}
        p2 = build_phase2_shortlist_candidates(
            p1, best_weights_map=best_map, vram=self.vram, shortlist_count=4, additional_epochs=25
        )
        self.assertEqual(len(p2), 4)
        # Top candidates get 960px resolution
        self.assertEqual(p2[0].imgsz, 960)
        self.assertEqual(p2[1].imgsz, 960)
        # Warm-started weights
        self.assertEqual(p2[0].weights, f"runs/detect/{p1[0].id}/weights/best.pt")
        self.assertEqual(p2[0].parent_candidate_id, p1[0].id)

    def test_phase3_finalist_builder(self):
        p1 = get_default_phase1_candidates(self.vram)
        best_map = {c.id: f"runs/detect/{c.id}/weights/best.pt" for c in p1}
        p2 = build_phase2_shortlist_candidates(p1, best_map, self.vram, shortlist_count=4)
        best_map_p2 = {c.id: f"runs/detect/{c.id}/weights/best.pt" for c in p2}
        p3 = build_phase3_finalist_candidates(p2, best_map_p2, self.vram, finalist_count=2)
        self.assertEqual(len(p3), 2)
        self.assertEqual(p3[0].phase, 3)
        self.assertEqual(p3[1].phase, 3)


class TestMetricsAndScoring(unittest.TestCase):
    def test_drone_annotation_score_priority(self):
        # Candidate A: High recall (0.90), moderate mAP (0.50)
        # Candidate B: Moderate recall (0.70), high mAP (0.70)
        # In drone annotation curation, Candidate A MUST win because missed objects cannot be curated!
        mA = CandidateMetrics(
            candidate_id="A", name="High Recall", phase=1, family="yolo11", scale="n",
            imgsz=640, batch=16, recall=0.90, small_object_recall=0.85, mAP50_95=0.50, precision=0.75,
            hard_set_recall=0.70
        )
        mB = CandidateMetrics(
            candidate_id="B", name="High mAP", phase=1, family="yolo11", scale="n",
            imgsz=640, batch=16, recall=0.70, small_object_recall=0.60, mAP50_95=0.70, precision=0.85,
            hard_set_recall=0.55
        )

        scoreA = compute_composite_drone_score(mA)
        scoreB = compute_composite_drone_score(mB)

        mA.composite_drone_score = scoreA
        mB.composite_drone_score = scoreB

        self.assertGreater(scoreA, scoreB, f"Candidate A (Recall {mA.recall}) should beat B (Recall {mB.recall})")
        self.assertEqual(round(mA.missed_object_rate, 1), 10.0)
        self.assertEqual(round(mB.missed_object_rate, 1), 30.0)


class TestHardDatasetAnalysis(unittest.TestCase):
    def test_annotation_analysis(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lbl_file = Path(tmpdir) / "001.txt"
            # 3 tiny boxes, 1 normal box
            lbl_file.write_text(
                "0 0.5 0.5 0.02 0.02\n"   # area 0.0004 (tiny)
                "0 0.3 0.3 0.015 0.015\n" # area 0.000225 (tiny)
                "0 0.7 0.7 0.03 0.03\n"   # area 0.0009 (tiny)
                "0 0.1 0.1 0.2 0.2\n"     # area 0.04 (normal)
            )
            n_box, n_tiny, ratio, min_a, med_a = analyze_annotation_file(lbl_file)
            self.assertEqual(n_box, 4)
            self.assertEqual(n_tiny, 3)
            self.assertEqual(ratio, 0.75)
            self.assertAlmostEqual(min_a, 0.000225)


class TestTournamentControllerMock(unittest.TestCase):
    def test_full_mock_tournament_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TournamentConfig(
                preset=TournamentPreset.SMOKE_TEST,
                output_dir=str(Path(tmpdir) / "runs" / "tournament"),
                total_budget_hours=0.2,
                phase1_candidates_count=2,
                phase2_shortlist_count=1,
                phase3_finalists_count=1,
                smoke_test=True,
                use_hard_val=False,
            )
            controller = TournamentController(config=cfg, mock_mode=True)
            res = controller.run()

            self.assertIsNotNone(res.recommended_winner)
            self.assertGreater(len(res.leaderboard), 0)
            self.assertTrue((Path(cfg.output_dir) / "state.json").exists())

            # Verify saved reports
            save_tournament_reports(
                leaderboard=res.leaderboard,
                winner=res.recommended_winner,
                top_finalists=res.top_finalists,
                total_elapsed_s=res.total_elapsed_seconds,
                config=cfg,
                output_dir=res.reports_dir,
                checkpoints_dir=res.checkpoints_dir,
            )

            self.assertTrue((res.reports_dir / "LEADERBOARD.md").exists())
            self.assertTrue((res.reports_dir / "leaderboard.json").exists())
            self.assertTrue((res.reports_dir / "leaderboard.csv").exists())
            self.assertTrue((res.reports_dir / "tournament_summary.txt").exists())

            # Verify state file content
            state_data = json.loads((Path(cfg.output_dir) / "state.json").read_text(encoding="utf-8"))
            self.assertIn("completed_candidates", state_data)
            self.assertIn("phase_results", state_data)


class TestPlatformUtils(unittest.TestCase):
    def test_python_detector(self):
        exe = get_project_python_exe()
        self.assertTrue(Path(exe).exists())

    def test_windows_sleep_inhibitor(self):
        # Entering and exiting context manager should never raise
        with WindowsSleepInhibitor(active=True):
            pass

    def test_windows_vt_mode(self):
        # Should execute safely on all platforms
        enable_windows_virtual_terminal()


if __name__ == "__main__":
    unittest.main()
