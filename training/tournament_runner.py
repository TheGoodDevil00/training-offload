"""
tournament_runner.py
====================
CLI and automated entrypoint for the Overnight Drone-Footage Model Tournament.
Supports headless overnight execution, quick smoke testing, and interactive TUI launch.

Usage:
    python training/tournament_runner.py --preset overnight_standard
    python training/tournament_runner.py --smoke-test
    python training/tournament_runner.py --hours 6.5
    python training/tournament_runner.py --tui
    python training/tournament_runner.py --mock
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from training.tournament.config import (
    TournamentConfig,
    TournamentPreset,
    get_preset_config,
)
from training.tournament.controller import TournamentController
from training.tournament.hard_dataset import build_hard_validation_dataset
from training.tournament.platform_utils import (
    WindowsSleepInhibitor,
    enable_windows_virtual_terminal,
    get_project_python_exe,
)
from training.tournament.reporter import format_time_duration, save_tournament_reports
from training.tournament.tui_tournament import (
    LiveTournamentMonitor,
    screen_view_leaderboard,
    tournament_main,
)


def parse_args():
    p = argparse.ArgumentParser(description="Overnight Drone-Footage Model Tournament")
    p.add_argument(
        "--preset",
        type=str,
        choices=["overnight_standard", "overnight_extended", "fast_screening", "smoke_test"],
        default="overnight_standard",
        help="Tournament preset configuration (default: overnight_standard)",
    )
    p.add_argument(
        "--hours",
        type=float,
        default=None,
        help="Override total overnight time budget in hours (e.g. 6.5)",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Execute a quick ~10-minute 2-candidate smoke test",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="Run mock simulation mode (verifies tournament pipeline and reporting without GPU)",
    )
    p.add_argument(
        "--tui",
        action="store_true",
        help="Launch the interactive Terminal User Interface (TUI) workstation",
    )
    p.add_argument(
        "--leaderboard",
        action="store_true",
        help="Display the existing tournament leaderboard and exit",
    )
    p.add_argument(
        "--build-hard-val",
        action="store_true",
        help="Generate or refresh the hard validation set and exit",
    )
    p.add_argument(
        "--output",
        type=str,
        default="runs/tournament",
        help="Output directory for tournament checkpoints and reports",
    )
    return p.parse_args()


def main():
    enable_windows_virtual_terminal()
    args = parse_args()

    # 1. TUI Interactive Mode
    if args.tui:
        tournament_main()
        return

    # 2. View Leaderboard Mode
    output_dir = PROJECT_ROOT / args.output
    if args.leaderboard:
        screen_view_leaderboard(output_dir)
        return

    # 3. Build Hard Validation Set Only
    if args.build_hard_val:
        data_yaml = PROJECT_ROOT / "datasets" / "usable" / "yolo-human" / "data.yaml"
        if not data_yaml.exists():
            print(f"[X] Dataset not found at {data_yaml}")
            sys.exit(1)
        build_hard_validation_dataset(data_yaml)
        print("[OK] Hard validation set built successfully.")
        return

    # 4. Tournament Execution
    preset_name = "smoke_test" if args.smoke_test else args.preset
    preset_enum = TournamentPreset(preset_name)
    config = get_preset_config(preset_enum)

    if args.hours is not None:
        config.total_budget_hours = args.hours

    config.output_dir = args.output

    print("=" * 72)
    print("  🏆 OVERNIGHT DRONE-FOOTAGE MODEL TOURNAMENT")
    print("=" * 72)
    print(f"  • Preset:            {config.preset.value.upper()}")
    print(f"  • Time Budget:       {config.total_budget_hours:.1f} hours ({format_time_duration(config.total_budget_seconds)})")
    print(f"  • Hardware Profile:  NVIDIA RTX 4050 6GB VRAM (Safe Batching)")
    print(f"  • Mode:              {'Mock Dry-Run' if args.mock else 'Live GPU Training'}")
    print(f"  • Priority Metric:   Recall (35%) + Small-Object (25%) + mAP50-95 (20%)")
    print(f"  • Phasing:           Phase 1 (Screen) -> Phase 2 (Shortlist) -> Phase 3 (Finalists)")
    print("=" * 72 + "\n")

    monitor = LiveTournamentMonitor()
    controller = TournamentController(config=config, callbacks=monitor, mock_mode=args.mock)

    # Inhibit Windows sleep while the tournament runs overnight
    with WindowsSleepInhibitor(active=True):
        try:
            result = controller.run()

            # Save reports, leaderboard, and training curves
            save_tournament_reports(
                leaderboard=result.leaderboard,
                winner=result.recommended_winner,
                top_finalists=result.top_finalists,
                total_elapsed_s=result.total_elapsed_seconds,
                config=config,
                output_dir=result.reports_dir,
                checkpoints_dir=result.checkpoints_dir,
            )

            print("\n" + "=" * 72)
            print(f"  ✔ TOURNAMENT COMPLETE ({format_time_duration(result.total_elapsed_seconds)})")
            if result.recommended_winner:
                w = result.recommended_winner
                print(f"  🥇 Winner:           {w.candidate_id} ({w.name})")
                print(f"  • Drone Score:      {w.composite_drone_score:.2f} / 100")
                print(f"  • Recall:           {w.recall * 100:.1f}% (Missed: {w.missed_object_rate:.1f}%)")
                print(f"  • Small-Obj Recall: {w.small_object_recall * 100:.1f}%")
                print(f"  • Best Weights:     {w.weights_path}")
            print(f"  • Detailed Report:  {result.reports_dir / 'LEADERBOARD.md'}")
            print("=" * 72)

        except KeyboardInterrupt:
            print("\n[!] Tournament paused by user. State preserved in state.json.")
            sys.exit(130)
        except Exception as e:
            print(f"\n[X] Tournament execution failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
