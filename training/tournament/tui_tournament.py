"""
tui_tournament.py
=================
Interactive Terminal User Interface for the Overnight Drone-Footage Model Tournament.
Tailored for Windows & Linux console environments, with live hardware telemetry,
dynamic leaderboard display, and real-time tournament progression tracking.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import (
    CandidateConfig,
    TournamentConfig,
    TournamentPreset,
    VRAMProfile,
    get_preset_config,
)
from .controller import TournamentCallbacks, TournamentController, TournamentResult
from .hard_dataset import build_hard_validation_dataset
from .metrics import CandidateMetrics
from .platform_utils import (
    WindowsSleepInhibitor,
    enable_windows_virtual_terminal,
    get_project_python_exe,
    read_key_universal,
)
from .reporter import format_time_duration

# Ensure Windows VT mode is active on module import
enable_windows_virtual_terminal()

PROJECT_ROOT = Path.cwd()

# --------------------------------------------------------------------------- #
# ANSI Styling Constants
# --------------------------------------------------------------------------- #
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
UNDERLINE = "\033[4m"
RESET = "\033[0m"

GREEN = "\033[38;5;82m"
BRIGHT_GREEN = "\033[38;5;120m"
CYAN = "\033[38;5;51m"
BLUE = "\033[38;5;75m"
YELLOW = "\033[38;5;220m"
RED = "\033[38;5;196m"
MAGENTA = "\033[38;5;207m"
WHITE = "\033[38;5;255m"
GRAY = "\033[38;5;244m"
DARK_GRAY = "\033[38;5;238m"

BG_DARK = "\033[48;5;236m"
BG_BLUE = "\033[48;5;24m"
BG_CYAN = "\033[48;5;30m"
INVERSE = "\033[7m"


def render_bar(pct: float, width: int = 12) -> str:
    """Renders a colored progress/utilization bar."""
    pct = max(0.0, min(100.0, pct))
    filled_len = int(round(width * (pct / 100.0)))
    empty_len = width - filled_len

    if pct >= 85.0:
        bar_color = RED
    elif pct >= 65.0:
        bar_color = YELLOW
    else:
        bar_color = GREEN

    filled = "█" * filled_len
    empty = "░" * empty_len
    return f"{bar_color}[{filled}{empty}]{RESET} {pct:4.1f}%"

def safe_wait_enter(prompt: str = "\nPress [Enter] to return..."):
    """Safely waits for Enter key, guarding against EOF in non-interactive/piped environments."""
    if sys.stdin.isatty():
        try:
            input(prompt)
        except (EOFError, KeyboardInterrupt):
            pass
    else:
        print("")

# --------------------------------------------------------------------------- #
# System Telemetry Engine
# --------------------------------------------------------------------------- #

_STATS_CACHE: Optional[Dict[str, Any]] = None
_STATS_TIME = 0.0


def get_system_stats(force: bool = False) -> Dict[str, Any]:
    """Queries CPU, RAM, Disk, and NVIDIA GPU telemetry with TTL caching."""
    global _STATS_CACHE, _STATS_TIME
    now = time.time()
    if not force and _STATS_CACHE is not None and (now - _STATS_TIME) < 2.0:
        return _STATS_CACHE

    stats: Dict[str, Any] = {
        "cpu_pct": 0.0,
        "cpu_str": "N/A",
        "cpu_temp": "N/A",
        "ram_str": "N/A",
        "ram_pct": 0.0,
        "disk_str": "N/A",
        "disk_pct": 0.0,
    }

    try:
        import psutil
        cpu_pct = psutil.cpu_percent(interval=None)
        stats["cpu_pct"] = cpu_pct
        stats["cpu_str"] = f"{cpu_pct:4.1f}%"

        vm = psutil.virtual_memory()
        used_gb = (vm.total - vm.available) / (1024 ** 3)
        total_gb = vm.total / (1024 ** 3)
        stats["ram_pct"] = vm.percent
        stats["ram_str"] = f"{used_gb:.1f} / {total_gb:.1f} GB"

        du = psutil.disk_usage(str(PROJECT_ROOT))
        d_used = du.used / (1024 ** 3)
        d_total = du.total / (1024 ** 3)
        stats["disk_pct"] = du.percent
        stats["disk_str"] = f"{d_used:.1f} / {d_total:.1f} GB"
    except Exception:
        pass

    # NVIDIA GPU Telemetry (nvidia-smi)
    smi_cmd = "nvidia-smi.exe" if os.name == "nt" else "nvidia-smi"
    try:
        res = subprocess.run(
            [smi_cmd, "--query-gpu=name,temperature.gpu,memory.used,memory.total,utilization.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=0.8
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            if len(parts) >= 6:
                g_name, g_temp, g_used, g_total, g_util, g_power = parts[:6]
                used_vram = float(g_used)
                total_vram = float(g_total)
                vram_pct = (used_vram / total_vram * 100) if total_vram else 0
                stats["gpu_name"] = g_name.replace("NVIDIA GeForce ", "")
                stats["gpu_temp"] = f"{g_temp}°C"
                stats["vram_pct"] = vram_pct
                stats["vram_str"] = f"{used_vram / 1024:.2f} / {total_vram / 1024:.2f} GB"
                stats["gpu_util"] = f"{g_util}%"
                stats["gpu_power"] = f"{float(g_power):.1f}W"
    except Exception:
        pass

    _STATS_CACHE = stats
    _STATS_TIME = now
    return stats


# --------------------------------------------------------------------------- #
# Terminal Canvas & Interactive Menu Engine
# --------------------------------------------------------------------------- #

class TerminalCanvas:
    """Manages alternate screen buffer, raw input, and clean restoration."""
    def __init__(self, alt_screen: bool = True):
        self.alt_screen = alt_screen
        self.old_termios = None

    def __enter__(self):
        enable_windows_virtual_terminal()
        if os.name != "nt" and sys.stdin.isatty():
            try:
                import termios
                import tty
                self.old_termios = termios.tcgetattr(sys.stdin.fileno())
                tty.setcbreak(sys.stdin.fileno())
            except Exception:
                pass

        if self.alt_screen:
            sys.stdout.write("\033[?1049h\033[2J\033[H\033[?25l")
        else:
            sys.stdout.write("\033[?25l")
        sys.stdout.flush()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.alt_screen:
            sys.stdout.write("\033[?25h\033[?1049l")
        else:
            sys.stdout.write("\033[?25h")
        sys.stdout.flush()

        if self.old_termios is not None:
            try:
                import termios
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.old_termios)
            except Exception:
                pass


def render_banner_lines(subtitle: str = "Sequential Successive-Halving Tournament") -> List[str]:
    """Renders the top banner and live telemetry bar."""
    stats = get_system_stats()
    lines: List[str] = []

    lines.append(f"{BOLD}{CYAN}╭────────────────────────────────────────────────────────────────────────╮{RESET}")
    lines.append(f"{BOLD}{CYAN}│     DRONE-FOOTAGE MODEL TOURNAMENT — Offline Annotation Workstation    │{RESET}")
    lines.append(f"{BOLD}{CYAN}│     {subtitle:<66} │{RESET}")
    lines.append(f"{BOLD}{CYAN}╰────────────────────────────────────────────────────────────────────────╯{RESET}")

    cpu_bar = render_bar(stats["cpu_pct"], width=10)
    ram_bar = render_bar(stats["ram_pct"], width=10)
    disk_bar = render_bar(stats["disk_pct"], width=10)

    lines.append(f"  {BOLD}Hardware Telemetry Dashboard:{RESET}")
    lines.append(f"    • {BOLD}CPU Usage:{RESET}    {cpu_bar} {stats['cpu_str']:<10} {DIM}|{RESET} {BOLD}RAM:{RESET} {ram_bar} {stats['ram_str']}")
    lines.append(f"    • {BOLD}Disk Space:{RESET}   {disk_bar} {stats['disk_str']} (workspace)")

    if "vram_str" in stats:
        vram_bar = render_bar(stats.get("vram_pct", 0), width=10)
        gpu_name = stats.get("gpu_name", "GPU")
        lines.append(f"    • {BOLD}GPU ({gpu_name}):{RESET} {vram_bar} {stats['vram_str']} {DIM}|{RESET} {stats.get('gpu_power', '')} {DIM}|{RESET} {YELLOW}{stats.get('gpu_temp', '')}{RESET}")

    lines.append(f"  {DIM}─ Target Spec: Laptop RTX 4050 (6 GB VRAM) | Metric: Recall (P1) & Small-Object ─{RESET}")
    lines.append(f"{DARK_GRAY}{'─' * 74}{RESET}")
    return lines


def prompt_select_menu(title: str, options: List[Dict[str, Any]], default_idx: int = 0) -> Optional[Any]:
    """Interactive zero-flicker menu with arrow navigation, Vi keys, and number selection."""
    if not sys.stdin.isatty():
        print(f"\n{BOLD}{title}{RESET}")
        for i, opt in enumerate(options, 1):
            print(f"  {i}. {opt.get('label', str(opt))}")
        print("  0. Back / Cancel")
        val = input(f"Select option [0-{len(options)}]: ").strip()
        if val == "0" or not val:
            return None
        try:
            idx = int(val) - 1
            if 0 <= idx < len(options):
                return options[idx].get("val", options[idx])
        except Exception:
            pass
        return None

    selected = default_idx
    with TerminalCanvas(alt_screen=True):
        while True:
            lines = []
            lines.extend(render_banner_lines())
            lines.append(f"  {BOLD}{WHITE}╭─ {title} ─{'─' * max(0, 58 - len(title))}╮{RESET}")
            lines.append(f"  {DIM}│  Use [↑/↓] or [j/k] to navigate, [Enter] or [1-{len(options)}] to select, [Esc/q/0] to cancel  │{RESET}")
            lines.append(f"  {BOLD}{WHITE}├────────────────────────────────────────────────────────────────────────┤{RESET}")

            for i, opt in enumerate(options):
                lbl = opt.get("label", str(opt))
                desc = opt.get("desc", "")
                is_active = (i == selected)
                key_num = str(i + 1) if i < 9 else "-"

                if is_active:
                    marker = f"{CYAN}▶{RESET}"
                    line_str = f"  {marker} {BG_BLUE}{BOLD}{WHITE} [{key_num}] {lbl:<34}{RESET}"
                    if desc:
                        line_str += f" {DIM}{CYAN}← {desc}{RESET}"
                else:
                    marker = " "
                    line_str = f"  {marker}   [{key_num}] {lbl:<34}"
                    if desc:
                        line_str += f" {DARK_GRAY}{desc}{RESET}"

                lines.append(line_str)

            lines.append(f"  {BOLD}{WHITE}╰────────────────────────────────────────────────────────────────────────╯{RESET}")
            lines.append(f"  {DIM}[Esc / q / 0] Back to previous menu{RESET}")
            lines.append("")

            buf = "\033[H" + "\r\n".join(lines) + "\r\n\033[J"
            sys.stdout.write(buf)
            sys.stdout.flush()

            key = read_key_universal(timeout_s=0.2)
            if key in ("UP", "k"):
                selected = (selected - 1) % len(options)
            elif key in ("DOWN", "j"):
                selected = (selected + 1) % len(options)
            elif key == "ENTER":
                return options[selected].get("val", options[selected])
            elif key in ("ESC", "q", "0"):
                return None
            elif key.isdigit():
                idx = int(key) - 1
                if 0 <= idx < len(options):
                    return options[idx].get("val", options[idx])


# --------------------------------------------------------------------------- #
# Live Tournament Monitor Callback Handler
# --------------------------------------------------------------------------- #

class LiveTournamentMonitor(TournamentCallbacks):
    """Real-time terminal dashboard updating as training proceeds."""

    def __init__(self):
        self.current_phase = 1
        self.active_candidate: Optional[CandidateConfig] = None
        self.candidate_index = 0
        self.total_candidates_in_phase = 0
        self.completed_metrics: List[CandidateMetrics] = []
        self.recent_logs: List[str] = []
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.total_budget_s = 6.5 * 3600.0

    def on_tournament_start(self, config: TournamentConfig):
        self.total_budget_s = config.total_budget_seconds
        self.start_time = time.time()
        self.on_log(f"Tournament initialized. Budget: {config.total_budget_hours:.1f} hours.")

    def on_phase_start(self, phase: int, candidates: List[CandidateConfig]):
        with self.lock:
            self.current_phase = phase
            self.total_candidates_in_phase = len(candidates)
            self.candidate_index = 0
        self.on_log(f"Started Phase {phase} with {len(candidates)} candidates.")

    def on_candidate_start(self, candidate: CandidateConfig, index: int, total: int):
        with self.lock:
            self.active_candidate = candidate
            self.candidate_index = index
            self.total_candidates_in_phase = total
        self.on_log(f"Candidate [{index}/{total}] {candidate.id} started ({candidate.imgsz}px, batch {candidate.batch}).")

    def on_candidate_complete(self, candidate: CandidateConfig, metrics: CandidateMetrics):
        with self.lock:
            self.completed_metrics.append(metrics)
            self.completed_metrics.sort(key=lambda m: m.composite_drone_score, reverse=True)
            for idx, m in enumerate(self.completed_metrics, 1):
                m.rank = idx
        self.on_log(
            f"✔ Completed {candidate.id}: Score {metrics.composite_drone_score:.1f} "
            f"(Recall: {metrics.recall*100:.1f}%, Small-Obj: {metrics.small_object_recall*100:.1f}%)"
        )

    def on_phase_complete(self, phase: int, ranked_metrics: List[CandidateMetrics]):
        top_name = ranked_metrics[0].candidate_id if ranked_metrics else "N/A"
        self.on_log(f"🏁 Phase {phase} finished. Top contender: {top_name}")

    def on_tournament_complete(self, leaderboard: List[CandidateMetrics], winner: Optional[CandidateMetrics]):
        winner_id = winner.candidate_id if winner else "N/A"
        self.on_log(f"🏆 TOURNAMENT COMPLETE! Recommended Winner: {winner_id}")

    def on_log(self, message: str, level: str = "info"):
        timestamp = time.strftime("%H:%M:%S")
        prefix = f"[{timestamp}]"
        if level == "warning":
            fmt_msg = f"{prefix} {YELLOW}⚠ {message}{RESET}"
        elif level == "error":
            fmt_msg = f"{prefix} {RED}✖ {message}{RESET}"
        else:
            fmt_msg = f"{prefix} {message}"

        with self.lock:
            self.recent_logs.append(fmt_msg)
            if len(self.recent_logs) > 6:
                self.recent_logs.pop(0)

        # Print directly so output streams in standard terminal
        print(f"  {fmt_msg}")


# --------------------------------------------------------------------------- #
# Menu Handlers & Screens
# --------------------------------------------------------------------------- #

def screen_view_leaderboard(output_dir: Path):
    """Displays the tournament leaderboard table in terminal."""
    state_file = output_dir / "state.json"
    leaderboard_csv = output_dir / "leaderboard.csv"

    if not state_file.exists() and not leaderboard_csv.exists():
        print("\n" + "=" * 70)
        print(f"  {YELLOW}No tournament results found under {output_dir.relative_to(PROJECT_ROOT)}{RESET}")
        print("  Execute a tournament run (Menu Option 1 or 2) to generate results.")
        print("=" * 70)
        input(f"\n{DIM}Press Enter to return to menu...{RESET}")
        return

    # Read state or CSV
    rows: List[Dict[str, Any]] = []
    if leaderboard_csv.exists():
        try:
            import csv
            with open(leaderboard_csv, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            pass

    print("\n" + "=" * 74)
    print(f"{BOLD}{CYAN}  🏆 TOURNAMENT LEADERBOARD (Ranked by Drone Annotation Score){RESET}")
    print("=" * 74)

    if rows:
        header_fmt = f"  {BOLD}{'#':<3} {'Candidate ID':<22} {'Phase':<6} {'ImgSz':<7} {'Recall':<9} {'SmallRec':<10} {'mAP50-95':<10} {'Score':<8}{RESET}"
        print(header_fmt)
        print(f"  {DARK_GRAY}{'─' * 70}{RESET}")
        for r in rows:
            rank = r.get("rank", "-")
            cid = r.get("candidate_id", "-")[:21]
            p = f"P{r.get('phase', '1')}"
            imgsz = f"{r.get('imgsz', '640')}px"
            try:
                rec = f"{float(r.get('recall', 0))*100:.1f}%"
                srec = f"{float(r.get('small_object_recall', 0))*100:.1f}%"
                m95 = f"{float(r.get('mAP50_95', 0))*100:.1f}%"
                score = f"{float(r.get('composite_drone_score', 0)):.1f}"
            except Exception:
                rec, srec, m95, score = "-", "-", "-", "-"

            medal = f"{YELLOW}🥇{RESET}" if rank == "1" else (f"{WHITE}🥈{RESET}" if rank == "2" else (f"{MAGENTA}🥉{RESET}" if rank == "3" else "  "))
            print(f" {medal}{rank:<2} {cid:<22} {p:<6} {imgsz:<7} {rec:<9} {srec:<10} {m95:<10} {BOLD}{GREEN}{score:<8}{RESET}")
    else:
        print(f"  {YELLOW}State found, but no finished candidate metrics yet.{RESET}")

    print("=" * 74)
    print(f"  Full documentation: {output_dir / 'LEADERBOARD.md'}")
    print(f"  Artifacts & plots:  {output_dir / 'training_curves.png'}")
    safe_wait_enter(f"\n{DIM}Press [Enter] to return to menu...{RESET}")


def screen_build_hard_dataset():
    """Generates or inspects the hard validation subset."""
    data_yaml = PROJECT_ROOT / "datasets" / "usable" / "yolo-human" / "data.yaml"
    if not data_yaml.exists():
        print(f"\n{RED}Error: Base dataset not found at {data_yaml.relative_to(PROJECT_ROOT)}{RESET}")
        print("Prepare the VisDrone dataset first via Main TUI.")
        safe_wait_enter("Press [Enter] to return...")
        return

    print("\n" + "=" * 70)
    print(f"{BOLD}{CYAN}  🎯 BUILD HARD VALIDATION SUBSET{RESET}")
    print("=" * 70)
    print("  Curating difficult frames: tiny objects, blur, distant crowds, shadows...")
    try:
        hard_yaml = build_hard_validation_dataset(
            data_yaml_path=data_yaml,
            max_frames=150,
            copy_files=(os.name == "nt"),  # Copy on Windows if symlink permissions restricted
        )
        print(f"\n{BOLD}{GREEN}✔ Hard validation set ready!{RESET}")
        print(f"  Config: {hard_yaml}")
    except Exception as e:
        print(f"\n{BOLD}{RED}✖ Failed to build hard validation set: {e}{RESET}")

    safe_wait_enter(f"\n{DIM}Press [Enter] to return to menu...{RESET}")

def screen_download_dataset():
    """Step 2: Download VisDrone raw dataset (~2.5 GB)."""
    data_dir = PROJECT_ROOT / "datasets" / "usable"
    train_dir = data_dir / "VisDrone2019-DET-train"
    val_dir = data_dir / "VisDrone2019-DET-val"
    if (train_dir / "images").exists() and (val_dir / "images").exists():
        print(f"\n{BOLD}{GREEN}✔ VisDrone dataset already downloaded and extracted at {data_dir.relative_to(PROJECT_ROOT)}.{RESET}")
        safe_wait_enter("Press [Enter] to return...")
        return

    print("\n" + "=" * 74)
    print(f"{BOLD}{CYAN}  📥 STEP 2: DOWNLOAD VISDRONE DATASET (~2.5 GB){RESET}")
    print("=" * 74)
    if os.name == "nt":
        cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(PROJECT_ROOT / "scripts" / "2_download_data.ps1")]
    else:
        cmd = ["bash", str(PROJECT_ROOT / "scripts" / "2_download_data.sh")]
    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    except Exception as e:
        print(f"\n{BOLD}{RED}✖ Download failed: {e}{RESET}")
    safe_wait_enter(f"\n{DIM}Press [Enter] to return to menu...{RESET}")


def screen_prepare_dataset():
    """Step 3: Convert VisDrone annotations to single-class YOLO dataset."""
    data_dir = PROJECT_ROOT / "datasets" / "usable"
    train_dir = data_dir / "VisDrone2019-DET-train"
    val_dir = data_dir / "VisDrone2019-DET-val"
    out_dir = data_dir / "yolo-human"
    yaml_file = out_dir / "data.yaml"

    if yaml_file.exists():
        print(f"\n{BOLD}{GREEN}✔ YOLO dataset is already prepared at {out_dir.relative_to(PROJECT_ROOT)}.{RESET}")
        ans = input("  Re-run conversion? (y/N): ").strip().lower()
        if ans != "y":
            return

    if not (train_dir / "annotations").exists() or not (val_dir / "annotations").exists():
        print(f"\n{BOLD}{RED}✖ Raw VisDrone folders not found.{RESET}")
        print("  Please run option '📥 Download VisDrone Dataset' first.")
        safe_wait_enter("Press [Enter] to return...")
        return

    print("\n" + "=" * 74)
    print(f"{BOLD}{CYAN}  📁 STEP 3: PREPARE YOLO DATASET (VisDrone -> single-class human){RESET}")
    print("=" * 74)
    py_exe = get_project_python_exe(PROJECT_ROOT)
    cmd = [
        py_exe,
        str(PROJECT_ROOT / "training" / "prepare_dataset.py"),
        "--train", str(train_dir),
        "--val", str(val_dir),
        "--out", str(out_dir),
        "--copy",
    ]
    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        print(f"\n{BOLD}{GREEN}✔ Dataset prepared successfully!{RESET} ({yaml_file.relative_to(PROJECT_ROOT)})")
    except Exception as e:
        print(f"\n{BOLD}{RED}✖ Preparation failed: {e}{RESET}")
    safe_wait_enter(f"\n{DIM}Press [Enter] to return to menu...{RESET}")


def screen_package_results():
    """Step 5: Export NCNN/ONNX, evaluate, and package deliverables into .zip."""
    print("\n" + "=" * 74)
    print(f"{BOLD}{CYAN}  📦 STEP 5: EXPORT, EVALUATE & PACKAGE RESULTS (.zip){RESET}")
    print("=" * 74)
    if os.name == "nt":
        cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(PROJECT_ROOT / "scripts" / "5_package_results.ps1")]
    else:
        cmd = ["bash", str(PROJECT_ROOT / "scripts" / "5_package_results.sh")]
    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    except Exception as e:
        print(f"\n{BOLD}{RED}✖ Packaging failed: {e}{RESET}")
    safe_wait_enter(f"\n{DIM}Press [Enter] to return to menu...{RESET}")


def run_tournament_workflow(preset: TournamentPreset, mock: bool = False):
    """Launches the sequential tournament with live telemetry and sleep prevention."""
    cfg = get_preset_config(preset)

    # Pre-flight check: ensure dataset exists before running real tournament
    data_yaml = PROJECT_ROOT / cfg.data_yaml
    if not mock and not data_yaml.exists():
        print(f"\n{BOLD}{RED}✖ Dataset not found at {data_yaml.relative_to(PROJECT_ROOT)}{RESET}")
        print("  Please download and prepare the dataset first using menu options:")
        print("    • '📥 Download VisDrone Dataset (~2.5 GB)'")
        print("    • '📁 Prepare YOLO Dataset'\n")
        safe_wait_enter("Press [Enter] to return...")
        return

    print("\n" + "=" * 74)
    print(f"{BOLD}{CYAN}  🚀 LAUNCHING DRONE MODEL TOURNAMENT ({preset.value.upper()}){RESET}")
    print("=" * 74)
    print(f"  • Time Budget:      {cfg.total_budget_hours:.1f} hours ({cfg.total_budget_seconds:.0f} seconds)")
    print(f"  • Hardware Profile: RTX 4050 6GB VRAM (batch safety auto-managed)")
    print(f"  • Target Metric:    Recall (35%) + Small-Object (25%) + mAP50-95 (20%)")
    print(f"  • Phasing:          P1: Screening -> P2: Shortlist -> P3: Finalists")
    print(f"  • Output Folder:    {cfg.output_dir}")
    print(f"  • Sleep Protection: Active (Windows SetThreadExecutionState)")
    print("=" * 74 + "\n")

    monitor = LiveTournamentMonitor()
    controller = TournamentController(config=cfg, callbacks=monitor, mock_mode=mock)

    # Inhibit Windows sleep while the tournament runs overnight
    with WindowsSleepInhibitor(active=True):
        try:
            res = controller.run()
            # Generate all final reports, leaderboard, and plots
            from .reporter import save_tournament_reports
            save_tournament_reports(
                leaderboard=res.leaderboard,
                winner=res.recommended_winner,
                top_finalists=res.top_finalists,
                total_elapsed_s=res.total_elapsed_seconds,
                config=cfg,
                output_dir=res.reports_dir,
                checkpoints_dir=res.checkpoints_dir,
            )

            print("\n" + "=" * 74)
            print(f"{BOLD}{GREEN}✔ TOURNAMENT COMPLETE ({format_time_duration(res.total_elapsed_seconds)}){RESET}")
            if res.recommended_winner:
                print(f"  Recommended Winner: {BOLD}{res.recommended_winner.candidate_id}{RESET}")
                print(f"  Drone Quality Score: {GREEN}{res.recommended_winner.composite_drone_score:.2f} / 100{RESET}")
                print(f"  Recall (Priority #1): {GREEN}{res.recommended_winner.recall*100:.1f}%{RESET}")
                print(f"  Small-Object Recall: {GREEN}{res.recommended_winner.small_object_recall*100:.1f}%{RESET}")
                print(f"  Best Weights:        {res.recommended_winner.weights_path}")
            print(f"  Detailed Report:     {res.reports_dir / 'LEADERBOARD.md'}")
            print("=" * 74)

        except KeyboardInterrupt:
            print(f"\n{YELLOW}⚠ Tournament paused by user. Progress saved to state.json.{RESET}")
        except Exception as e:
            print(f"\n{RED}✖ Tournament error: {e}{RESET}")

    safe_wait_enter(f"\n{DIM}Press [Enter] to return to menu...{RESET}")


# --------------------------------------------------------------------------- #
# Main Interactive Tournament Workstation Loop
# --------------------------------------------------------------------------- #

def tournament_main():
    """Main interactive entrypoint for the Tournament TUI."""
    output_dir = PROJECT_ROOT / "runs" / "tournament"

    options = [
        {
            "label": "🚀 Launch Standard Overnight Tournament",
            "desc": "Step 4.5: 6.5 hours, 8 candidates, 3 phases, adaptive GPU budgeting",
            "val": "overnight_standard",
        },
        {
            "label": "⚡ Quick Smoke Test Tournament",
            "desc": "10-15 minutes, 2 candidates @ 320px, validates GPU & pipeline end-to-end",
            "val": "smoke_test",
        },
        {
            "label": "⏱️ Fast Screening Tournament",
            "desc": "2.5 hours, 6 candidates, fast successive-halving filter",
            "val": "fast_screening",
        },
        {
            "label": "📥 Download VisDrone Dataset (~2.5 GB)",
            "desc": "Step 2: Downloads raw train + val zip splits from GitHub releases",
            "val": "download_dataset",
        },
        {
            "label": "📁 Prepare YOLO Dataset",
            "desc": "Step 3: Converts VisDrone annotations into YOLO format (data.yaml)",
            "val": "prepare_dataset",
        },
        {
            "label": "📦 Package Final Deliverables (.zip)",
            "desc": "Step 5: Exports NCNN/ONNX, evaluates mAP, and zips training-results.zip",
            "val": "package_results",
        },
        {
            "label": "📊 View Tournament Leaderboard & Finalists",
            "desc": "Inspect ranked metrics table, winners, and training curves",
            "val": "view_leaderboard",
        },
        {
            "label": "🎯 Build / Inspect Hard Validation Set",
            "desc": "Curate tiny-object, blur, distant crowd validation frames",
            "val": "hard_dataset",
        },
        {
            "label": "🧪 Mock Dry-Run Tournament (Test Verification)",
            "desc": "Simulate 3-phase tournament execution with synthetic metrics",
            "val": "mock_run",
        },
    ]

    while True:
        choice = prompt_select_menu("Overnight Drone Model Tournament — Workstation", options)

        if choice == "overnight_standard":
            run_tournament_workflow(TournamentPreset.OVERNIGHT_STANDARD, mock=False)
        elif choice == "smoke_test":
            run_tournament_workflow(TournamentPreset.SMOKE_TEST, mock=False)
        elif choice == "fast_screening":
            run_tournament_workflow(TournamentPreset.FAST_SCREENING, mock=False)
        elif choice == "download_dataset":
            screen_download_dataset()
        elif choice == "prepare_dataset":
            screen_prepare_dataset()
        elif choice == "package_results":
            screen_package_results()
        elif choice == "view_leaderboard":
            screen_view_leaderboard(output_dir)
        elif choice == "hard_dataset":
            screen_build_hard_dataset()
        elif choice == "mock_run":
            run_tournament_workflow(TournamentPreset.SMOKE_TEST, mock=True)
        elif choice is None:
            break


if __name__ == "__main__":
    tournament_main()
