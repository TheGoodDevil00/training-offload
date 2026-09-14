#!/usr/bin/env python3
"""
tui.py
======
Unified Terminal User Interface (TUI) for the Rescue Swarm Human Detector pipeline.

Consolidates all dataset preparation, model training, edge export, performance evaluation,
video inference, dataset health diagnostics, and training metrics inspection into a single
interactive terminal workstation with real-time hardware telemetry and RPi 5 simulation.

Usage:
    python training/tui.py
"""

import os
import sys
import shutil
import subprocess
import threading
import time
import select
from pathlib import Path

# Add project root to sys.path and set working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from training.tournament.platform_utils import (
    enable_windows_virtual_terminal,
    read_key_universal,
)
enable_windows_virtual_terminal()


def get_python_exe():
    """Detects project venv python (.venv or venv), falling back to sys.executable."""
    if os.name == "nt":
        candidates = [
            PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
            PROJECT_ROOT / "venv" / "Scripts" / "python.exe",
        ]
    else:
        candidates = [
            PROJECT_ROOT / ".venv" / "bin" / "python",
            PROJECT_ROOT / "venv" / "bin" / "python",
        ]
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return sys.executable


# Auto re-exec into project venv if available and not already in use
_target_py = get_python_exe()
if _target_py != sys.executable and Path(_target_py).exists() and not os.environ.get("SWARM_VENV_REEXEC"):
    os.environ["SWARM_VENV_REEXEC"] = "1"
    try:
        os.execv(_target_py, [_target_py] + sys.argv)
    except Exception:
        pass

DATASET_DIR = PROJECT_ROOT / "datasets" / "usable" / "yolo-human"
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
LAST_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "last.pt"
BASE_WEIGHTS = PROJECT_ROOT / "yolo11n.pt"

YOLO26_DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "train_yolo26" / "weights" / "best.pt"
YOLO26_LAST_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "train_yolo26" / "weights" / "last.pt"
YOLO26_BASE_WEIGHTS = PROJECT_ROOT / "yolo26n.pt"


def get_default_model_path() -> str:
    """Returns the most relevant model checkpoint path available."""
    if YOLO26_DEFAULT_WEIGHTS.exists():
        return str(YOLO26_DEFAULT_WEIGHTS.relative_to(PROJECT_ROOT))
    if DEFAULT_WEIGHTS.exists():
        return str(DEFAULT_WEIGHTS.relative_to(PROJECT_ROOT))
    # Check any runs/detect/*/weights/best.pt
    pts = list((PROJECT_ROOT / "runs" / "detect").glob("*/weights/best.pt"))
    if pts:
        return str(pts[0].relative_to(PROJECT_ROOT))
    if YOLO26_BASE_WEIGHTS.exists():
        return str(YOLO26_BASE_WEIGHTS.relative_to(PROJECT_ROOT))
    if BASE_WEIGHTS.exists():
        return str(BASE_WEIGHTS.relative_to(PROJECT_ROOT))
    return "yolo26n.pt"
# ANSI Terminal Styling Constants
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
UNDERLINE = "\033[4m"
RESET = "\033[0m"

# Foreground Colors
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

# Background Highlights
BG_DARK = "\033[48;5;236m"
BG_BLUE = "\033[48;5;24m"
BG_CYAN = "\033[48;5;30m"
INVERSE = "\033[7m"


def render_bar(pct: float, width: int = 12) -> str:
    """Renders a high-resolution colored progress/utilization bar."""
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


# --------------------------------------------------------------------------- #
# System Telemetry & Hardware Monitoring Engine
# --------------------------------------------------------------------------- #

_STATS_CACHE = None
_STATS_TIME = 0.0
_PIPELINE_CACHE = None
_PIPELINE_TIME = 0.0


def get_system_stats(force: bool = False):
    """Queries current CPU, RAM, Disk, GPU, VRAM, and thermal sensors with TTL caching."""
    global _STATS_CACHE, _STATS_TIME
    now = time.time()
    if not force and _STATS_CACHE is not None and (now - _STATS_TIME) < 2.5:
        return _STATS_CACHE

    stats = {
        "cpu_pct": 0.0,
        "cpu_str": "N/A",
        "cpu_temp": "N/A",
        "ram_str": "N/A",
        "ram_pct": 0.0,
        "disk_str": "N/A",
        "disk_pct": 0.0,
    }
    # 1. CPU & RAM & Disk via psutil if available
    try:
        import psutil
        # CPU
        cpu_pct = psutil.cpu_percent(interval=None)
        load1, _, _ = os.getloadavg()
        stats["cpu_pct"] = cpu_pct
        stats["cpu_str"] = f"{cpu_pct:4.1f}% (load: {load1:.2f})"

        # RAM
        vm = psutil.virtual_memory()
        used_gb = (vm.total - vm.available) / (1024 ** 3)
        total_gb = vm.total / (1024 ** 3)
        stats["ram_pct"] = vm.percent
        stats["ram_str"] = f"{used_gb:.1f} / {total_gb:.1f} GB"

        # Disk (Project Root)
        du = psutil.disk_usage(str(PROJECT_ROOT))
        d_used = du.used / (1024 ** 3)
        d_total = du.total / (1024 ** 3)
        stats["disk_pct"] = du.percent
        stats["disk_str"] = f"{d_used:.1f} / {d_total:.1f} GB"

        # CPU Temp via psutil sensors
        try:
            temps = psutil.sensors_temperatures()
            for key in ("k10temp", "coretemp", "cpu_thermal", "acpitz"):
                if key in temps and temps[key]:
                    stats["cpu_temp"] = f"{temps[key][0].current:.1f}°C"
                    break
        except Exception:
            pass
    except Exception:
        # Fallback to /proc
        try:
            with open("/proc/meminfo", "r") as f:
                mem_lines = f.readlines()
            mem_info = {}
            for line in mem_lines:
                parts = line.split(":")
                if len(parts) == 2:
                    mem_info[parts[0].strip()] = int(parts[1].split()[0])
            total_mb = mem_info.get("MemTotal", 0) // 1024
            avail_mb = mem_info.get("MemAvailable", 0) // 1024
            used_mb = total_mb - avail_mb
            pct = (used_mb / total_mb * 100) if total_mb else 0
            stats["ram_pct"] = pct
            stats["ram_str"] = f"{used_mb / 1024:.1f} / {total_mb / 1024:.1f} GB"
        except Exception:
            pass

    # CPU Temperature Fallback
    if stats["cpu_temp"] == "N/A":
        try:
            cpu_temps = []
            for zone in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
                try:
                    t = int(zone.read_text().strip()) / 1000.0
                    if 20 <= t <= 110:
                        cpu_temps.append(t)
                except Exception:
                    pass
            if cpu_temps:
                stats["cpu_temp"] = f"{max(cpu_temps):.1f}°C"
        except Exception:
            pass

    # 2. NVIDIA GPU Stats (nvidia-smi)
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,temperature.gpu,memory.used,memory.total,utilization.gpu,power.draw",
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

class TelemetryThread(threading.Thread):
    """Background daemon updating the terminal window title bar with live stats."""
    def __init__(self, interval=2.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            stats = get_system_stats()
            vram_str = f" | VRAM: {stats['vram_str']}" if "vram_str" in stats else ""
            gpu_str = f" | GPU: {stats['gpu_temp']}" if "gpu_temp" in stats else ""
            title = f"Rescue Swarm Pipeline | CPU: {stats.get('cpu_str', 'N/A')} | RAM: {stats.get('ram_str', 'N/A')}{gpu_str}{vram_str}"
            try:
                sys.stdout.write(f"\033]0;{title}\007")
                sys.stdout.flush()
            except Exception:
                pass
            self.stop_event.wait(self.interval)

    def stop(self):
        self.stop_event.set()
        try:
            sys.stdout.write("\033]0;\007")
            sys.stdout.flush()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Pipeline State & Workspace Introspection
# --------------------------------------------------------------------------- #

def get_pipeline_state(force: bool = False):
    """Gathers status across dataset, training checkpoints, exports, and media with TTL caching."""
    global _PIPELINE_CACHE, _PIPELINE_TIME
    now = time.time()
    if not force and _PIPELINE_CACHE is not None and (now - _PIPELINE_TIME) < 4.0:
        return _PIPELINE_CACHE

    state = {}
    raw_train = PROJECT_ROOT / "datasets" / "usable" / "VisDrone2019-DET-train"
    raw_val = PROJECT_ROOT / "datasets" / "usable" / "VisDrone2019-DET-val"
    raw_ok = (raw_train / "images").exists() and (raw_val / "images").exists()
    state["raw_ok"] = raw_ok
    state["raw_desc"] = "Downloaded (~2.5 GB)" if raw_ok else "Not downloaded (Step 2)"

    data_yaml = DATASET_DIR / "data.yaml"
    train_dir = DATASET_DIR / "train" / "images"
    val_dir = DATASET_DIR / "val" / "images"
    if data_yaml.exists() and train_dir.exists() and val_dir.exists():
        try:
            n_train = len(os.listdir(train_dir))
            n_val = len(os.listdir(val_dir))
            state["dataset_ok"] = True
            state["dataset_desc"] = f"Ready ({n_train:,} train / {n_val:,} val images)"
            state["n_train"] = n_train
            state["n_val"] = n_val
        except Exception:
            state["dataset_ok"] = True
            state["dataset_desc"] = f"Ready ({DATASET_DIR.relative_to(PROJECT_ROOT)})"
    else:
        state["dataset_ok"] = False
        state["dataset_desc"] = "Not prepared (Step 3)"
    # 2. Model Checkpoint Status
    runs_dir = PROJECT_ROOT / "runs" / "detect"
    checkpoints = []
    if runs_dir.exists():
        for r_dir in sorted(runs_dir.iterdir(), key=lambda d: d.stat().st_mtime if d.is_dir() else 0, reverse=True):
            if not r_dir.is_dir():
                continue
            best_file = r_dir / "weights" / "best.pt"
            csv_file = r_dir / "results.csv"
            if best_file.exists():
                checkpoints.append((r_dir.name, best_file, csv_file))

    if checkpoints:
        state["model_ok"] = True
        latest_name, latest_best, latest_csv = checkpoints[0]
        model_type_label = "YOLO26n" if "yolo26" in latest_name.lower() else ("YOLO11n" if latest_name == "train" else latest_name)
        train_info = f"Found {model_type_label} ({latest_name}/best.pt)"

        if latest_csv.exists():
            try:
                import csv
                with open(latest_csv, mode="r") as f:
                    rows = list(csv.DictReader(f))
                if rows:
                    headers = {k.strip(): k for k in rows[0].keys()}
                    m50_key = headers.get("metrics/mAP50(B)")
                    m95_key = headers.get("metrics/mAP50-95(B)")
                    ep_key = headers.get("epoch")
                    best_row = max(rows, key=lambda r: float(r[m50_key]) if r.get(m50_key) else 0)
                    m50_val = float(best_row[m50_key]) * 100
                    m95_val = float(best_row[m95_key]) * 100
                    ep_val = best_row[ep_key].strip()
                    total_ep = len(rows)
                    train_info = f"[{model_type_label}] Best mAP50: {m50_val:.1f}% | mAP50-95: {m95_val:.1f}% (ep {ep_val}/{total_ep})"
                    state["best_map50"] = m50_val
                    state["best_map95"] = m95_val
                    state["epochs_done"] = total_ep
            except Exception:
                pass
        if len(checkpoints) > 1:
            train_info += f" ({len(checkpoints)} runs ready)"
        state["model_desc"] = train_info
    else:
        state["model_ok"] = False
        state["model_desc"] = "Not trained yet"
    # 3. Edge Exports
    # 2b. Tournament Winner Checkpoints
    tourn_checkpoints = list((PROJECT_ROOT / "runs" / "tournament" / "checkpoints").glob("*/best.pt"))
    if tourn_checkpoints:
        state["tournament_ok"] = True
        state["tournament_desc"] = f"Winner ready ({len(tourn_checkpoints)} models)"
    elif (PROJECT_ROOT / "runs" / "tournament" / "state.json").exists():
        state["tournament_ok"] = True
        state["tournament_desc"] = "In progress / paused"
    else:
        state["tournament_ok"] = False
        state["tournament_desc"] = "Ready to launch"
    exports = list((PROJECT_ROOT / "runs").glob("**/*_ncnn_model")) + list((PROJECT_ROOT / "runs").glob("**/*.onnx"))
    state["exports"] = exports
    if exports:
        fmt_list = []
        for exp in exports:
            if "_ncnn_model" in exp.name:
                size_mb = sum(f.stat().st_size for f in exp.glob("*")) / (1024 * 1024)
                fmt_list.append(f"NCNN ({size_mb:.1f} MB)")
            elif exp.suffix == ".onnx":
                size_mb = exp.stat().st_size / (1024 * 1024)
                fmt_list.append(f"ONNX ({size_mb:.1f} MB)")
        state["export_desc"] = ", ".join(fmt_list)
    else:
        state["export_desc"] = "None"

    # 4. Input Media & Webcams
    videos = list(PROJECT_ROOT.glob("*.mp4")) + list(PROJECT_ROOT.glob("*.avi")) + list(PROJECT_ROOT.glob("*.mkv"))
    webcams = list(Path("/dev").glob("video*"))
    state["videos"] = [str(v.relative_to(PROJECT_ROOT)) for v in videos]
    state["webcams"] = [str(w) for w in webcams]

    # 5. Packaged Deliverables
    result_zips = sorted(PROJECT_ROOT.glob("training-results-*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    state["package_ok"] = bool(result_zips)
    state["package_desc"] = result_zips[0].name if result_zips else "Not packaged yet"

    _PIPELINE_CACHE = state
    _PIPELINE_TIME = now
    return state


def discover_models():
    """Finds all available PyTorch, NCNN, and ONNX models in workspace."""
    models = []
    seen_paths = set()

    # 1. Best / Last weights across runs/detect
    runs_dir = PROJECT_ROOT / "runs" / "detect"
    if runs_dir.exists():
        for run_dir in sorted(runs_dir.iterdir(), key=lambda d: d.stat().st_mtime if d.is_dir() else 0, reverse=True):
            if not run_dir.is_dir():
                continue
            best = run_dir / "weights" / "best.pt"
            last = run_dir / "weights" / "last.pt"
            run_name = run_dir.name
            tag = "YOLO26n" if "yolo26" in run_name.lower() else ("YOLO11n" if run_name == "train" else run_name)

            if best.exists() and best not in seen_paths:
                seen_paths.add(best)
                size_mb = best.stat().st_size / (1024 * 1024)
                mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(best.stat().st_mtime))
                models.append({
                    "path": str(best.relative_to(PROJECT_ROOT)),
                    "type": f"PyTorch Checkpoint - Best Val mAP ({tag})",
                    "size": f"{size_mb:.1f} MB",
                    "time": mtime,
                    "recommended": True,
                })

            if last.exists() and last not in seen_paths:
                seen_paths.add(last)
                size_mb = last.stat().st_size / (1024 * 1024)
                mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(last.stat().st_mtime))
                models.append({
                    "path": str(last.relative_to(PROJECT_ROOT)),
                    "type": f"PyTorch Checkpoint - Last Epoch ({tag})",
                    "size": f"{size_mb:.1f} MB",
                    "time": mtime,
                    "recommended": False,
                })
    # 1b. Tournament Winning Checkpoints
    tourn_dir = PROJECT_ROOT / "runs" / "tournament" / "checkpoints"
    if tourn_dir.exists():
        for cand_dir in sorted(tourn_dir.iterdir(), key=lambda d: d.stat().st_mtime if d.is_dir() else 0, reverse=True):
            if not cand_dir.is_dir():
                continue
            best_t = cand_dir / "best.pt"
            if best_t.exists() and best_t not in seen_paths:
                seen_paths.add(best_t)
                size_mb = best_t.stat().st_size / (1024 * 1024)
                mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(best_t.stat().st_mtime))
                models.append({
                    "path": str(best_t.relative_to(PROJECT_ROOT)),
                    "type": f"🏆 Tournament Winner Checkpoint ({cand_dir.name})",
                    "size": f"{size_mb:.1f} MB",
                    "time": mtime,
                    "recommended": True,
                })

    # 2. Exported NCNN models
    for ncnn_dir in (PROJECT_ROOT / "runs").glob("**/*_ncnn_model"):
        if ncnn_dir not in seen_paths:
            seen_paths.add(ncnn_dir)
            size_mb = sum(f.stat().st_size for f in ncnn_dir.glob("*")) / (1024 * 1024)
            mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(ncnn_dir.stat().st_mtime))
            models.append({
                "path": str(ncnn_dir.relative_to(PROJECT_ROOT)),
                "type": "NCNN Model (RPi 5 ARM NEON Optimized)",
                "size": f"{size_mb:.1f} MB",
                "time": mtime,
                "recommended": True,
            })

    # 3. ONNX models
    for onnx_file in (PROJECT_ROOT / "runs").glob("**/*.onnx"):
        if onnx_file not in seen_paths:
            seen_paths.add(onnx_file)
            size_mb = onnx_file.stat().st_size / (1024 * 1024)
            mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(onnx_file.stat().st_mtime))
            models.append({
                "path": str(onnx_file.relative_to(PROJECT_ROOT)),
                "type": "ONNX Runtime Model",
                "size": f"{size_mb:.1f} MB",
                "time": mtime,
                "recommended": False,
            })

    # 4. Base pretrained models
    if YOLO26_BASE_WEIGHTS.exists() and YOLO26_BASE_WEIGHTS not in seen_paths:
        size_mb = YOLO26_BASE_WEIGHTS.stat().st_size / (1024 * 1024)
        models.append({
            "path": str(YOLO26_BASE_WEIGHTS.relative_to(PROJECT_ROOT)),
            "type": "Pretrained Base YOLO26n (COCO-80, NMS-Free)",
            "size": f"{size_mb:.1f} MB",
            "time": "-",
            "recommended": False,
        })

    if BASE_WEIGHTS.exists() and BASE_WEIGHTS not in seen_paths:
        size_mb = BASE_WEIGHTS.stat().st_size / (1024 * 1024)
        models.append({
            "path": str(BASE_WEIGHTS.relative_to(PROJECT_ROOT)),
            "type": "Pretrained Base YOLO11n (COCO-80)",
            "size": f"{size_mb:.1f} MB",
            "time": "-",
            "recommended": False,
        })

    return models


# --------------------------------------------------------------------------- #
# Interactive Keyboard Navigation Engine
# --------------------------------------------------------------------------- #

def parse_escape_seq(seq: str) -> str:
    """Decodes ANSI, VT100, xterm, Kitty, and DECCKM escape sequences."""
    if not seq:
        return "ESC"
    # Standard ANSI & Application Cursor Keys (DECCKM / Kitty / tmux)
    if seq in ("[A", "OA") or seq.endswith("A"):
        return "UP"
    if seq in ("[B", "OB") or seq.endswith("B"):
        return "DOWN"
    if seq in ("[C", "OC") or seq.endswith("C"):
        return "RIGHT"
    if seq in ("[D", "OD") or seq.endswith("D"):
        return "LEFT"
    # Kitty keyboard protocol (CSI unicode-key u)
    if "57416" in seq:
        return "UP"
    if "57417" in seq:
        return "DOWN"
    if "57418" in seq:
        return "RIGHT"
    if "57419" in seq:
        return "LEFT"
    # Page Up / Down / Home / End
    if "5~" in seq:
        return "PAGE_UP"
    if "6~" in seq:
        return "PAGE_DOWN"
    if "H" in seq or "1~" in seq:
        return "HOME"
    if "F" in seq or "4~" in seq:
        return "END"
    return "ESC"


def read_key(fd: int) -> str:
    """Reads a keypress, supporting Windows and POSIX."""
    if os.name == "nt":
        return read_key_universal(timeout_s=0.05)
    try:
        raw_byte = os.read(fd, 1)
    except Exception:
        return ""
    if not raw_byte:
        return ""

    ch = raw_byte.decode("utf-8", errors="ignore")
    if ch == "\x1b":
        seq = ""
        while True:
            r, _, _ = select.select([fd], [], [], 0.02)
            if not r:
                break
            chunk = os.read(fd, 32).decode("utf-8", errors="ignore")
            if not chunk:
                break
            seq += chunk
        return parse_escape_seq(seq)
    elif ch in ("\r", "\n"):
        return "ENTER"
    elif ch in ("\x7f", "\x08"):
        return "BACKSPACE"
    elif ch == "\x03":  # Ctrl+C
        raise KeyboardInterrupt
    return ch


class TerminalSession:
    """Manages terminal cbreak mode, alternate screen buffer, and cursor visibility."""
    def __init__(self, alt_screen: bool = True):
        self.fd = sys.stdin.fileno() if sys.stdin.isatty() else None
        self.old_settings = None
        self.alt_screen = alt_screen

    def __enter__(self):
        if os.name != "nt" and self.fd is not None:
            try:
                import termios
                import tty
                self.old_settings = termios.tcgetattr(self.fd)
                tty.setcbreak(self.fd)
            except Exception:
                pass
        # Enter alternate screen buffer & hide cursor for a crisp, isolated canvas
        if self.alt_screen:
            sys.stdout.write("\033[?1049h\033[2J\033[H\033[?25l")
        else:
            sys.stdout.write("\033[?25l")
        sys.stdout.flush()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Exit alternate screen buffer & restore cursor
        if self.alt_screen:
            sys.stdout.write("\033[?25h\033[?1049l")
        else:
            sys.stdout.write("\033[?25h")
        sys.stdout.flush()
        if os.name != "nt" and self.fd is not None and self.old_settings is not None:
            try:
                import termios
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
            except Exception:
                pass
def prompt_select_menu(title: str, options: list, default_idx: int = 0):
    """
    Renders an interactive menu allowing navigation with Arrow keys (Up/Down),
    Vi keys (k/j), direct number keys (1-9), and Enter to confirm with zero flicker.
    """
    if not sys.stdin.isatty():
        # Fallback to plain prompt in non-interactive environment
        print(f"\n{BOLD}{title}{RESET}")
        for i, opt in enumerate(options, 1):
            lbl = opt.get("label", str(opt))
            print(f"  {i}. {lbl}")
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
    with TerminalSession(alt_screen=True) as session:
        while True:
            # Build screen buffer in memory (sub-millisecond)
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
                    line_str = f"  {marker} {BG_BLUE}{BOLD}{WHITE} [{key_num}] {lbl:<32}{RESET}"
                    if desc:
                        line_str += f" {DIM}{CYAN}← {desc}{RESET}"
                else:
                    marker = " "
                    line_str = f"  {marker}   [{key_num}] {lbl:<32}"
                    if desc:
                        line_str += f" {DARK_GRAY}{desc}{RESET}"

                lines.append(line_str)

            lines.append(f"  {BOLD}{WHITE}╰────────────────────────────────────────────────────────────────────────╯{RESET}")
            lines.append(f"  {DIM}[Esc / q / 0] Back to previous menu{RESET}")
            lines.append("")
            # Atomic render at cursor home with explicit \r\n to prevent any staircasing
            buf = "\033[H" + "\r\n".join(lines) + "\r\n\033[J"
            sys.stdout.write(buf)
            sys.stdout.flush()

            key = read_key(session.fd)

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

def prompt_string(label: str, default: str = "") -> str:
    prompt = f"  {BOLD}{label}{RESET} [{CYAN}{default}{RESET}]: " if default else f"  {BOLD}{label}{RESET}: "
    val = input(prompt).strip()
    return val if val else default


def prompt_int(label: str, default: int) -> int:
    val = prompt_string(label, str(default))
    try:
        return int(val)
    except ValueError:
        print(f"  {RED}Invalid integer. Using default {default}{RESET}")
        return default


def prompt_float(label: str, default: float) -> float:
    val = prompt_string(label, str(default))
    try:
        return float(val)
    except ValueError:
        print(f"  {RED}Invalid float. Using default {default}{RESET}")
        return default


# --------------------------------------------------------------------------- #
# Smart Asset & Media Pickers
# --------------------------------------------------------------------------- #

def pick_model(title: str = "Select Model Checkpoint", default_path: str = None) -> str:
    """Interactively scans and lets the user pick from discovered models."""
    models = discover_models()
    options = []
    target_default = default_path or get_default_model_path()

    default_idx = 0
    for idx, m in enumerate(models):
        rec_tag = f" {GREEN}(Recommended){RESET}" if m.get("recommended") else ""
        options.append({
            "label": m["path"],
            "desc": f"{m['type']} [{m['size']}]{rec_tag}",
            "val": m["path"],
        })
        if m["path"] == target_default or str(PROJECT_ROOT / m["path"]) == target_default:
            default_idx = idx

    options.append({
        "label": "Custom Path...",
        "desc": "Specify another file path or URL",
        "val": "__CUSTOM__",
    })

    chosen = prompt_select_menu(title, options, default_idx=default_idx)
    if not chosen:
        return None
    if chosen == "__CUSTOM__":
        return prompt_string("Enter model file/folder path", target_default)
    return chosen


def pick_media(title: str = "Select Input Media / Stream", default_src: str = "0") -> str:
    """Interactively scans workspace and webcam devices for inference/evaluation."""
    state = get_pipeline_state()
    options = []

    # Discovered Videos
    for vid in state["videos"]:
        try:
            sz_mb = (PROJECT_ROOT / vid).stat().st_size / (1024 * 1024)
            sz_str = f"{sz_mb:.1f} MB"
        except Exception:
            sz_str = "Video file"
        options.append({
            "label": vid,
            "desc": f"Local workspace drone video [{sz_str}]",
            "val": vid
        })

    # Webcams
    if state["webcams"]:
        for cam in state["webcams"]:
            idx = cam.replace("/dev/video", "")
            options.append({
                "label": f"Webcam {cam}",
                "desc": f"Live USB/V4L2 camera (source '{idx}')",
                "val": idx
            })
    else:
        options.append({
            "label": "Webcam (default: 0)",
            "desc": "Standard default video capture device index",
            "val": "0"
        })

    options.append({
        "label": "Custom Video Path / RTSP Stream...",
        "desc": "Specify video filename or network stream URL",
        "val": "__CUSTOM__"
    })

    chosen = prompt_select_menu(title, options)
    if not chosen:
        return None
    if chosen == "__CUSTOM__":
        return prompt_string("Enter video path, index, or RTSP URL", default_src)
    return chosen


# --------------------------------------------------------------------------- #
# UI Displays, Header, & Execution Wrapper
# --------------------------------------------------------------------------- #

def render_banner_lines() -> list:
    """Generates the banner, telemetry dashboard, and pipeline stepper as a list of lines."""
    stats = get_system_stats()
    state = get_pipeline_state()
    lines = []

    # 1. Header Box
    lines.append(f"{BOLD}{CYAN}╭────────────────────────────────────────────────────────────────────────╮{RESET}")
    lines.append(f"{BOLD}{CYAN}│     RESCUE SWARM: Aerial Human Detector (YOLO26 / YOLO11) - TUI        │{RESET}")
    lines.append(f"{BOLD}{CYAN}╰────────────────────────────────────────────────────────────────────────╯{RESET}")

    # 2. Live Hardware Telemetry Dashboard
    cpu_bar = render_bar(stats["cpu_pct"], width=10)
    ram_bar = render_bar(stats["ram_pct"], width=10)
    disk_bar = render_bar(stats["disk_pct"], width=10)

    lines.append(f"  {BOLD}Hardware Telemetry Dashboard:{RESET}")
    lines.append(f"    • {BOLD}CPU Usage:{RESET}    {cpu_bar} {stats['cpu_str']:<16} {DIM}|{RESET} {BOLD}Temp:{RESET} {YELLOW}{stats['cpu_temp']}{RESET}")
    lines.append(f"    • {BOLD}RAM Memory:{RESET}   {ram_bar} {stats['ram_str']}")
    lines.append(f"    • {BOLD}Disk Space:{RESET}   {disk_bar} {stats['disk_str']} (workspace)")

    if "vram_str" in stats:
        vram_bar = render_bar(stats.get("vram_pct", 0), width=10)
        gpu_name = stats.get("gpu_name", "GPU")
        lines.append(f"    • {BOLD}GPU ({gpu_name}):{RESET} {vram_bar} {stats['vram_str']} {DIM}|{RESET} {stats.get('gpu_power', '')} {DIM}|{RESET} {YELLOW}{stats.get('gpu_temp', '')}{RESET}")

    # 3. Target Deployment Reference
    lines.append(f"  {DIM}─ Target Edge Spec: Raspberry Pi 5 (4x Cortex-A76 @ 2.4GHz, 4GB RAM, Target: ≥15 FPS) ─{RESET}")

    # 4. Pipeline Lifecycle Stepper
    lines.append("")
    lines.append(f"  {BOLD}Pipeline Lifecycle Status:{RESET}")
    # Step 0: Download
    raw_ico = f"{GREEN}✔{RESET}" if state.get("raw_ok") else f"{YELLOW}•{RESET}"
    raw_txt = f"{GREEN}{state.get('raw_desc')}{RESET}" if state.get("raw_ok") else f"{YELLOW}{state.get('raw_desc')}{RESET}"
    lines.append(f"    {raw_ico} {BOLD}[0. Download Data]{RESET} {raw_txt}")

    # Step 1: Prep
    prep_ico = f"{GREEN}✔{RESET}" if state["dataset_ok"] else f"{YELLOW}•{RESET}"
    prep_txt = f"{GREEN}{state['dataset_desc']}{RESET}" if state["dataset_ok"] else f"{YELLOW}{state['dataset_desc']}{RESET}"
    lines.append(f"    {prep_ico} {BOLD}[1. Prep Dataset ]{RESET} {prep_txt}")

    # Step 2: Model
    model_ico = f"{GREEN}✔{RESET}" if state["model_ok"] else f"{YELLOW}•{RESET}"
    model_txt = f"{GREEN}{state['model_desc']}{RESET}" if state["model_ok"] else f"{YELLOW}{state['model_desc']}{RESET}"
    lines.append(f"    {model_ico} {BOLD}[2. Train Model  ]{RESET} {model_txt}")

    # Step T: Tournament
    tourn_ico = f"{GREEN}✔{RESET}" if state.get("tournament_ok") else f"{DIM}•{RESET}"
    tourn_txt = f"{GREEN}{state.get('tournament_desc', 'Ready')}{RESET}" if state.get("tournament_ok") else f"{DIM}{state.get('tournament_desc', 'Ready')}{RESET}"
    lines.append(f"    {tourn_ico} {BOLD}[T. Tournament   ]{RESET} {tourn_txt}")

    # Step 3: Export
    exp_ico = f"{GREEN}✔{RESET}" if state["exports"] else f"{DIM}•{RESET}"
    exp_txt = f"{GREEN}{state['export_desc']}{RESET}" if state["exports"] else f"{DIM}Pending export{RESET}"
    lines.append(f"    {exp_ico} {BOLD}[3. Edge Export  ]{RESET} {exp_txt}")

    # Step P: Package
    pkg_ico = f"{GREEN}✔{RESET}" if state.get("package_ok") else f"{DIM}•{RESET}"
    pkg_txt = f"{GREEN}{state.get('package_desc')}{RESET}" if state.get("package_ok") else f"{DIM}{state.get('package_desc')}{RESET}"
    lines.append(f"    {pkg_ico} {BOLD}[P. Package Zip  ]{RESET} {pkg_txt}")

    # Step 4: Inference
    infer_ready = f"{GREEN}Ready ({len(state['videos'])} video(s), {len(state['webcams'])} camera(s)){RESET}"
    lines.append(f"    {GREEN}✔{RESET} {BOLD}[4. Inference    ]{RESET} {infer_ready}")
    lines.append(f"{DARK_GRAY}{'─' * 74}{RESET}")
    return lines


def print_banner():
    """Prints status header and hardware telemetry with explicit CRLF."""
    lines = render_banner_lines()
    sys.stdout.write("\033[2J\033[H" + "\r\n".join(lines) + "\r\n")
    sys.stdout.flush()

def run_command(cmd_list: list, task_name: str = "Pipeline Task"):
    """Executes a command array with live terminal output and hardware snapshots."""
    cmd_str = " ".join(cmd_list)
    print(f"\n{BOLD}{CYAN}╭─ Executing {task_name} ─{'─' * max(0, 52 - len(task_name))}╮{RESET}")
    print(f"  {BOLD}Command:{RESET} {GREEN}{cmd_str}{RESET}")

    start_stats = get_system_stats()
    gpu_snapshot = f" | VRAM: {start_stats['vram_str']} | GPU: {start_stats['gpu_temp']}" if "vram_str" in start_stats else ""
    print(f"  {DIM}[Hardware Start] CPU: {start_stats['cpu_str']} | RAM: {start_stats['ram_str']}{gpu_snapshot}{RESET}")
    print(f"{BOLD}{CYAN}╰────────────────────────────────────────────────────────────────────────╯{RESET}\n")

    monitor = TelemetryThread(interval=2.5)
    monitor.start()
    t0 = time.perf_counter()

    try:
        res = subprocess.run(cmd_list, cwd=PROJECT_ROOT)
        monitor.stop()
        elapsed = time.perf_counter() - t0

        print(f"\n{DARK_GRAY}{'─' * 74}{RESET}")
        end_stats = get_system_stats()
        end_gpu = f" | VRAM: {end_stats['vram_str']} | GPU: {end_stats['gpu_temp']}" if "vram_str" in end_stats else ""
        print(f"{DIM}[Hardware Finish in {elapsed:.1f}s] CPU: {end_stats['cpu_str']} | RAM: {end_stats['ram_str']}{end_gpu}{RESET}")

        if res.returncode == 0:
            print(f"\n{BOLD}{GREEN}✔ {task_name} completed successfully! ({elapsed:.1f}s){RESET}")
        else:
            print(f"\n{BOLD}{RED}✖ {task_name} failed with exit code {res.returncode}.{RESET}")
    except KeyboardInterrupt:
        monitor.stop()
        print(f"\n{BOLD}{YELLOW}⚠ Process interrupted by user.{RESET}")
    except Exception as e:
        monitor.stop()
        print(f"\n{BOLD}{RED}✖ Error launching command: {e}{RESET}")

    get_system_stats(force=True)
    get_pipeline_state(force=True)
    input(f"\n{DIM}Press [Enter] to return to menu...{RESET}")

# --------------------------------------------------------------------------- #
# Menu Handlers
# --------------------------------------------------------------------------- #

def menu_download_dataset():
    """Step 2: Download the VisDrone2019-DET raw dataset (~2.5 GB)."""
    data_dir = PROJECT_ROOT / "datasets" / "usable"
    train_dir = data_dir / "VisDrone2019-DET-train"
    val_dir = data_dir / "VisDrone2019-DET-val"
    if (train_dir / "images").exists() and (val_dir / "images").exists():
        print(f"\n{BOLD}{GREEN}✔ VisDrone dataset is already downloaded and extracted at {data_dir.relative_to(PROJECT_ROOT)}.{RESET}")
        input(f"\n{DIM}Press [Enter] to return to menu...{RESET}")
        return

    if os.name == "nt":
        cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(PROJECT_ROOT / "scripts" / "2_download_data.ps1")]
    else:
        cmd = ["bash", str(PROJECT_ROOT / "scripts" / "2_download_data.sh")]
    run_command(cmd, "Download VisDrone Dataset (~2.5 GB)")


def menu_package_results():
    """Step 5: Export NCNN/ONNX, evaluate, and package deliverables into .zip."""
    if os.name == "nt":
        cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(PROJECT_ROOT / "scripts" / "5_package_results.ps1")]
    else:
        cmd = ["bash", str(PROJECT_ROOT / "scripts" / "5_package_results.sh")]
    run_command(cmd, "Package Deliverables (.zip)")


def check_dependencies(py_exe: str = None) -> tuple:
    """Checks if essential packages (PIL, cv2, yaml, torch, ultralytics) are importable."""
    exe = py_exe or get_python_exe()
    check_code = (
        "import sys\n"
        "missing = []\n"
        "for mod, name in [('PIL', 'Pillow'), ('cv2', 'opencv-python'), ('yaml', 'PyYAML'), ('torch', 'PyTorch'), ('ultralytics', 'Ultralytics')]:\n"
        "    try:\n"
        "        __import__(mod)\n"
        "    except ImportError:\n"
        "        missing.append(name)\n"
        "if missing:\n"
        "    print(','.join(missing))\n"
        "    sys.exit(1)\n"
    )
    try:
        res = subprocess.run([exe, "-c", check_code], capture_output=True, text=True)
        if res.returncode != 0:
            return False, res.stdout.strip() or "Core dependencies missing"
        return True, ""
    except Exception as e:
        return False, str(e)


def menu_install_dependencies():
    """Step 1 / Repair: Installs project requirements.txt into the active venv."""
    req_file = PROJECT_ROOT / "requirements.txt"
    if not req_file.exists():
        print(f"\n{BOLD}{RED}✖ requirements.txt not found at {req_file}{RESET}")
        input(f"\n{DIM}Press [Enter] to return...{RESET}")
        return
    py = get_python_exe()
    cmd = [py, "-m", "pip", "install", "-r", str(req_file)]
    run_command(cmd, "Install / Repair Dependencies")


def ensure_dependencies() -> bool:
    """Verifies dependencies, prompting to install them if missing."""
    ok, missing = check_dependencies()
    if not ok:
        print(f"\n{BOLD}{YELLOW}⚠ Missing Python dependencies in environment: {missing}{RESET}")
        print("  These packages are required to run this step.")
        ans = input(f"  Install missing requirements now? (Y/n): ").strip().lower()
        if ans != "n":
            menu_install_dependencies()
            ok_after, _ = check_dependencies()
            return ok_after
        return False
    return True


def menu_prepare_dataset():
    """Step 3: Dataset conversion and options (VisDrone -> single-class human)."""
    if not ensure_dependencies():
        return
    raw_train = PROJECT_ROOT / "datasets" / "usable" / "VisDrone2019-DET-train"
    raw_val = PROJECT_ROOT / "datasets" / "usable" / "VisDrone2019-DET-val"
    if not (raw_train / "images").exists() or not (raw_val / "images").exists():
        print(f"\n{BOLD}{RED}✖ Raw VisDrone folders not found under datasets/usable/.{RESET}")
        print("  Please download the dataset first (Step 2).")
        if input("  Download VisDrone dataset now? (Y/n): ").strip().lower() != "n":
            menu_download_dataset()
            if not (raw_train / "images").exists():
                return
        else:
            return

    options = [
        {"label": "Copy Images (Recommended for Windows)", "desc": "Physically copy images to avoid symlink permission errors (--copy)", "val": "copy"},
        {"label": "Standard Symlink", "desc": "Keep all frames (incl. negatives), symlink images to save disk", "val": "default"},
        {"label": "Only Frames with Humans", "desc": "Filter out non-human negative frames (--only-with-humans)", "val": "humans_only"},
        {"label": "Custom Split Paths", "desc": "Specify custom train/val/out directories", "val": "custom"},
    ]

    choice = prompt_select_menu("Step 3: Prepare VisDrone Dataset", options)
    if not choice:
        return

    cmd = [get_python_exe(), "training/prepare_dataset.py"]

    if choice == "humans_only":
        cmd.append("--only-with-humans")
    elif choice == "copy":
        cmd.append("--copy")
    elif choice == "custom":
        train_path = prompt_string("Train split dir", "datasets/usable/VisDrone2019-DET-train")
        val_path = prompt_string("Val split dir", "datasets/usable/VisDrone2019-DET-val")
        out_path = prompt_string("Output dir", "datasets/usable/yolo-human")
        cmd.extend(["--train", train_path, "--val", val_path, "--out", out_path])
        if input("  Copy images instead of symlink? (y/N): ").lower().startswith("y"):
            cmd.append("--copy")
        if input("  Only keep images with humans? (y/N): ").lower().startswith("y"):
            cmd.append("--only-with-humans")

    run_command(cmd, "Dataset Preparation")


def menu_train():
    """Step 2: Model training workflows (supports YOLO26n and YOLO11n)."""
    if not ensure_dependencies():
        return
    arch_options = [
        {
            "label": "🚀 YOLO26n (Recommended — End-to-End, 5.3 GFLOPs)",
            "desc": "Dual-assignment head, NMS-free, optimized for 4 GB RTX 3050 & RPi 5",
            "val": "yolo26",
        },
        {
            "label": "⚡ YOLO11n (Baseline — 6.6 GFLOPs)",
            "desc": "Standard baseline drone human detector (runs training/train.py)",
            "val": "yolo11",
        },
    ]

    arch = prompt_select_menu("Step 2: Select Model Architecture to Train", arch_options)
    if not arch:
        return

    is_yolo26 = arch == "yolo26"
    script_name = "training/train_yolo26.py" if is_yolo26 else "training/train.py"
    default_base = "yolo26n.pt" if is_yolo26 else "yolo11n.pt"
    default_best = YOLO26_DEFAULT_WEIGHTS if is_yolo26 else DEFAULT_WEIGHTS
    default_last = YOLO26_LAST_WEIGHTS if is_yolo26 else LAST_WEIGHTS
    arch_label = "YOLO26n" if is_yolo26 else "YOLO11n"

    options = [
        {
            "label": f"Stage 1: Frozen Backbone (Recommended for {arch_label})",
            "desc": "100 epochs, imgsz 512, freeze 10, batch 16 (tuned for 4 GB RTX 3050)",
            "val": "stage1",
        },
        {
            "label": "Stage 2: Full Fine-Tune",
            "desc": "Unfreeze backbone (freeze 0), cosine LR, 40 epochs, batch 16",
            "val": "stage2",
        },
        {
            "label": "Quick Smoke Test",
            "desc": "3 epochs, 0.2 dataset fraction, batch 16, profiling",
            "val": "smoke",
        },
        {
            "label": "Resume Interrupted Run",
            "desc": f"Resume from {default_last.relative_to(PROJECT_ROOT) if default_last.exists() else default_last.name}",
            "val": "resume",
        },
        {
            "label": "Custom Training Run",
            "desc": "Configure epochs, batch, imgsz, freeze, LR, optimizer manually",
            "val": "custom",
        },
        {
            "label": "🏆 Overnight Drone Model Tournament",
            "desc": "Sequential successive-halving across YOLO11/26 architectures",
            "val": "tournament",
        },
    ]

    choice = prompt_select_menu(f"Step 2: Train Model ({arch_label})", options)
    if not choice:
        return

    if choice == "tournament":
        run_command([get_python_exe(), "training/tournament_runner.py", "--tui"], "Overnight Model Tournament")
        return

    cmd = [get_python_exe(), script_name]

    if choice == "stage1":
        cmd.extend([
            "--epochs", "100",
            "--batch", "16",
            "--imgsz", "512",
            "--freeze", "10",
            "--single-cls",
        ])
    elif choice == "stage2":
        default_pt = str(default_best.relative_to(PROJECT_ROOT)) if default_best.exists() else default_base
        weights_in = pick_model(f"Select Base Checkpoint for Stage 2 ({arch_label})", default_pt)
        if not weights_in:
            return
        stage2_name = f"train_{arch}_stage2" if is_yolo26 else "train_stage2"
        cmd.extend([
            "--freeze", "0",
            "--epochs", "40",
            "--batch", "16",
            "--imgsz", "512",
            "--single-cls",
            "--cos-lr",
            "--weights", weights_in,
            "--name", stage2_name,
        ])
    elif choice == "smoke":
        smoke_name = f"smoke_{arch}"
        cmd.extend([
            "--epochs", "3",
            "--batch", "16",
            "--fraction", "0.2",
            "--profile",
            "--name", smoke_name,
        ])
    elif choice == "resume":
        if not default_last.exists():
            last_rel = default_last.relative_to(PROJECT_ROOT) if default_last.is_relative_to(PROJECT_ROOT) else default_last
            print(f"\n{RED}Error: No last.pt checkpoint found at {last_rel}{RESET}")
            input("Press Enter to continue...")
            return
        if is_yolo26:
            cmd.extend([
                "--weights", str(default_last),
                "--resume",
            ])
        else:
            cmd.extend([
                "--weights", str(default_last),
                "--epochs", "100",
                "--batch", "-1",
                "--imgsz", "512",
                "--single-cls",
            ])
    elif choice == "custom":
        epochs = prompt_int("Epochs", 100)
        imgsz = prompt_int("Input resolution (px, divisible by 32)", 512)
        batch = prompt_int("Batch size (16 recommended for 4 GB RTX 3050, -1 for auto)", 16)
        freeze = prompt_int("Frozen layers (10 = backbone, 0 = all)", 10)
        weights = pick_model(f"Weights checkpoint ({arch_label})", default_base)
        if not weights:
            return
        cmd.extend([
            "--epochs", str(epochs),
            "--imgsz", str(imgsz),
            "--batch", str(batch),
            "--freeze", str(freeze),
            "--weights", weights,
        ])
        if input("  Enable Cosine LR schedule? (Y/n): ").lower() != "n":
            cmd.append("--cos-lr")

    run_command(cmd, f"Model Training ({arch_label})")

def menu_export():
    """Step 3: Edge export and quantization."""
    options = [
        {"label": "NCNN FP16 (ARM NEON - Recommended)", "desc": "Fastest runtime on RPi 5 Cortex-A76 (15-25+ FPS target)", "val": "ncnn_fp16"},
        {"label": "NCNN INT8 (Quantized)", "desc": "Calibrated int8 quantization using data.yaml", "val": "ncnn_int8"},
        {"label": "ONNX INT8 (Alternative)", "desc": "Optimized ONNX graph with int8 quantization", "val": "onnx_int8"},
        {"label": "Custom Export", "desc": "Manually select format, resolution, and precision flags", "val": "custom"},
    ]

    choice = prompt_select_menu("Step 3: Edge Export & Quantization (RPi 5 Target)", options)
    if not choice:
        return

    model_path = pick_model("Select Checkpoint to Export", get_default_model_path())
    if not model_path:
        return
    cmd = [get_python_exe(), "training/export.py", "--model", model_path]

    if choice == "ncnn_fp16":
        cmd.extend(["--format", "ncnn", "--imgsz", "416", "--half"])
    elif choice == "ncnn_int8":
        data_path = prompt_string("Calibration data.yaml", "datasets/usable/yolo-human/data.yaml")
        cmd.extend(["--format", "ncnn", "--imgsz", "416", "--int8", "--data", data_path])
    elif choice == "onnx_int8":
        data_path = prompt_string("Calibration data.yaml", "datasets/usable/yolo-human/data.yaml")
        cmd.extend(["--format", "onnx", "--imgsz", "416", "--int8", "--data", data_path])
    elif choice == "custom":
        fmt = prompt_string("Format (ncnn / onnx)", "ncnn")
        imgsz = prompt_int("Input resolution (px)", 416)
        cmd.extend(["--format", fmt, "--imgsz", str(imgsz)])
        quant = prompt_string("Quantization (half / int8 / none)", "half").lower()
        if quant == "half":
            cmd.append("--half")
        elif quant == "int8":
            data_path = prompt_string("Calibration data.yaml", "datasets/usable/yolo-human/data.yaml")
            cmd.extend(["--int8", "--data", data_path])

    run_command(cmd, "Edge Model Export")


def menu_evaluate():
    """Step 4: Raspberry Pi 5 hardware simulation and benchmarking."""
    options = [
        {"label": "Speed Benchmark (Resolution & Format Sweep)", "desc": "Sweeps 640, 416, 352 with simulated 4 CPU threads & Pi 5 factor", "val": "speed"},
        {"label": "Accuracy Benchmark (mAP50 / mAP50-95)", "desc": "Evaluates detection precision/recall/mAP on validation split", "val": "accuracy"},
        {"label": "Quick 10-Frame RPi 5 Compliance Check", "desc": "Instant verification of >= 15 FPS target on movie.mp4", "val": "quick_check"},
    ]

    choice = prompt_select_menu("Step 4: RPi 5 Hardware Simulation & Evaluation", options)
    if not choice:
        return

    cmd = [get_python_exe(), "training/eval/evaluate.py"]

    if choice == "speed":
        cmd.append("speed")
        model_path = pick_model("Select Model to Benchmark", get_default_model_path())
        if not model_path:
            return
        video_path = pick_media("Select Benchmark Video", "movie.mp4")
        if not video_path:
            return
        resolutions = prompt_string("Resolution sweep (comma-separated)", "640,416,352")
        threads = prompt_int("Simulated RPi 5 CPU threads", 4)
        frames = prompt_int("Frames to benchmark", 60)

        cmd.extend(["--models", model_path, "--video", video_path,
                    "--imgsz", resolutions, "--threads", str(threads), "--frames", str(frames)])
    elif choice == "accuracy":
        cmd.append("accuracy")
        model_path = pick_model("Select Model Checkpoint (.pt)", get_default_model_path())
        if not model_path:
            return
        imgsz = prompt_int("Validation resolution (px)", 416)
        threads = prompt_int("Simulated CPU threads", 4)
        cmd.extend(["--model", model_path, "--imgsz", str(imgsz), "--threads", str(threads)])
    elif choice == "quick_check":
        cmd.append("speed")
        # Try finding NCNN model first, otherwise PT
        ncnn_models = list((PROJECT_ROOT / "runs").glob("**/*_ncnn_model"))
        target_model = str(ncnn_models[0].relative_to(PROJECT_ROOT)) if ncnn_models else get_default_model_path()
        target_vid = "movie.mp4" if (PROJECT_ROOT / "movie.mp4").exists() else pick_media("Select Video")
        if not target_vid:
            return
        cmd.extend(["--models", target_model, "--video", target_vid, "--imgsz", "416", "--frames", "15", "--threads", "4"])

    run_command(cmd, "Hardware Evaluation")


def menu_inference():
    """Step 5: Run real-time detection on video or webcam."""
    model_path = pick_model("Select Detection Model (.pt or exported NCNN/ONNX)", get_default_model_path())
    if not model_path:
        return
    input_src = pick_media("Select Input Video Source or Camera", "movie.mp4")
    if not input_src:
        return
    output_path = prompt_string("Output annotated video path", "output_annotated.mp4")
    imgsz = prompt_int("Inference resolution (px, matching export)", 416)
    conf = prompt_float("Confidence threshold", 0.35)
    cmd = [
        get_python_exe(), "training/inference.py",
        "--model", model_path,
        "--input", input_src,
        "--output", output_path,
        "--imgsz", str(imgsz),
        "--conf", str(conf)
    ]

    run_command(cmd, "Video Inference")


def menu_inspect_metrics():
    """Step 6: Training Metrics & Run History Inspector."""
    runs_dir = PROJECT_ROOT / "runs" / "detect"
    available_runs = []
    if runs_dir.exists():
        for r_dir in sorted(runs_dir.iterdir(), key=lambda d: d.stat().st_mtime if d.is_dir() else 0, reverse=True):
            if not r_dir.is_dir():
                continue
            csv_file = r_dir / "results.csv"
            if csv_file.exists():
                tag = "YOLO26n" if "yolo26" in r_dir.name.lower() else ("YOLO11n" if r_dir.name == "train" else r_dir.name)
                available_runs.append((r_dir, tag))

    if not available_runs:
        print_banner()
        print(f"{BOLD}{WHITE}╭─ Training Run Metrics & History Inspector ─────────────────────────────╮{RESET}")
        print(f"  {YELLOW}No training run results found under runs/detect/*/results.csv{RESET}")
        print(f"  Execute Step 2 (Model Training) to generate training results.")
        print(f"{BOLD}{WHITE}╰────────────────────────────────────────────────────────────────────────╯{RESET}")
        input(f"\n{DIM}Press Enter to return...{RESET}")
        return

    target_run_dir = available_runs[0][0]
    if len(available_runs) > 1:
        run_options = [
            {
                "label": f"Run: {r_dir.name} ({tag})",
                "desc": f"Results at {r_dir.relative_to(PROJECT_ROOT)}/results.csv",
                "val": r_dir,
            }
            for r_dir, tag in available_runs
        ]
        chosen = prompt_select_menu("Select Training Run to Inspect", run_options)
        if not chosen:
            return
        target_run_dir = chosen

    csv_path = target_run_dir / "results.csv"
    args_yaml = target_run_dir / "args.yaml"
    run_name = target_run_dir.name
    run_tag = "YOLO26n" if "yolo26" in run_name.lower() else ("YOLO11n" if run_name == "train" else run_name)

    print_banner()
    print(f"{BOLD}{WHITE}╭─ Training Run Metrics & History Inspector: {CYAN}{run_name}{WHITE} ({run_tag}) ─╮{RESET}")
    try:
        import csv
        with open(csv_path, mode="r") as f:
            rows = list(csv.DictReader(f))

        headers = {k.strip(): k for k in rows[0].keys()}
        m50_key = headers.get("metrics/mAP50(B)")
        m95_key = headers.get("metrics/mAP50-95(B)")
        mp_key = headers.get("metrics/precision(B)")
        mr_key = headers.get("metrics/recall(B)")
        tbox_key = headers.get("train/box_loss")
        tcls_key = headers.get("train/cls_loss")
        vbox_key = headers.get("val/box_loss")
        vcls_key = headers.get("val/cls_loss")
        ep_key = headers.get("epoch")

        total_epochs = len(rows)
        best_row = max(rows, key=lambda r: float(r[m50_key]) if r.get(m50_key) else 0)
        last_row = rows[-1]

        print(f"  {BOLD}Training Run Summary:{RESET}")
        print(f"    • Total Epochs Trained: {CYAN}{total_epochs}{RESET}")
        print(f"    • Best Epoch:           {GREEN}Epoch {best_row[ep_key].strip()}{RESET}")
        print(f"    • Best mAP50:           {GREEN}{float(best_row[m50_key])*100:.2f}%{RESET}")
        print(f"    • Best mAP50-95:        {GREEN}{float(best_row[m95_key])*100:.2f}%{RESET}")
        print(f"    • Precision (Best):     {float(best_row[mp_key])*100:.2f}%")
        print(f"    • Recall (Best):        {float(best_row[mr_key])*100:.2f}%")
        print(f"  {DIM}────────────────────────────────────────────────────────────────────────{RESET}")
        print(f"  {BOLD}Final Epoch Losses (Epoch {last_row[ep_key].strip()}):{RESET}")
        print(f"    • Train Loss: Box: {float(last_row[tbox_key]):.3f} | Cls: {float(last_row[tcls_key]):.3f}")
        print(f"    • Val Loss:   Box: {float(last_row[vbox_key]):.3f} | Cls: {float(last_row[vcls_key]):.3f}")

        # Show Hyperparameters if args.yaml exists
        if args_yaml.exists():
            try:
                import yaml
                with open(args_yaml) as yf:
                    meta = yaml.safe_load(yf)
                print(f"  {DIM}────────────────────────────────────────────────────────────────────────{RESET}")
                print(f"  {BOLD}Run Hyperparameters:{RESET}")
                print(f"    • Base Model: {meta.get('model', 'yolo11n.pt')} | ImgSz: {meta.get('imgsz')} | Batch: {meta.get('batch')}")
                print(f"    • Freeze Backbone: {meta.get('freeze')} layers | Cosine LR: {meta.get('cos_lr')} | Single Class: {meta.get('single_cls')}")
            except Exception:
                pass

        # Progress Sample Table (Last 5 epochs)
        print(f"  {DIM}────────────────────────────────────────────────────────────────────────{RESET}")
        print(f"  {BOLD}Recent Epoch Progression:{RESET}")
        print(f"  {'Epoch':<8}{'mAP50':<12}{'mAP50-95':<12}{'Precision':<12}{'Recall':<12}{'Val Box Loss':<12}")
        for r in rows[-5:]:
            print(f"  {r[ep_key].strip():<8}"
                  f"{float(r[m50_key])*100:>5.1f}%      "
                  f"{float(r[m95_key])*100:>5.1f}%      "
                  f"{float(r[mp_key])*100:>5.1f}%      "
                  f"{float(r[mr_key])*100:>5.1f}%      "
                  f"{float(r[vbox_key]):>6.3f}")

    except Exception as e:
        print(f"  {RED}Error reading training results: {e}{RESET}")

    print(f"{BOLD}{WHITE}╰────────────────────────────────────────────────────────────────────────╯{RESET}")
    input(f"\n{DIM}Press Enter to return...{RESET}")


def menu_dataset_health():
    """Step 7: Dataset Health & Integrity Check."""
    print_banner()
    print(f"{BOLD}{WHITE}╭─ Dataset Health & Integrity Verification ──────────────────────────────╮{RESET}")

    data_yaml = DATASET_DIR / "data.yaml"
    if not data_yaml.exists():
        print(f"  {RED}Dataset not found at {DATASET_DIR.relative_to(PROJECT_ROOT)}{RESET}")
        print(f"  Run Step 1 to prepare the VisDrone YOLO dataset.")
        print(f"{BOLD}{WHITE}╰────────────────────────────────────────────────────────────────────────╯{RESET}")
        input(f"\n{DIM}Press Enter to return...{RESET}")
        return

    train_img = DATASET_DIR / "train" / "images"
    train_lbl = DATASET_DIR / "train" / "labels"
    val_img = DATASET_DIR / "val" / "images"
    val_lbl = DATASET_DIR / "val" / "labels"

    n_tr_img = len(list(train_img.glob("*"))) if train_img.exists() else 0
    n_tr_lbl = len(list(train_lbl.glob("*"))) if train_lbl.exists() else 0
    n_va_img = len(list(val_img.glob("*"))) if val_img.exists() else 0
    n_va_lbl = len(list(val_lbl.glob("*"))) if val_lbl.exists() else 0

    print(f"  {BOLD}Split Integrity:{RESET}")
    tr_match = f"{GREEN}Matched ({n_tr_img:,} pairs){RESET}" if (n_tr_img == n_tr_lbl and n_tr_img > 0) else f"{RED}Mismatch ({n_tr_img} imgs vs {n_tr_lbl} lbls){RESET}"
    va_match = f"{GREEN}Matched ({n_va_img:,} pairs){RESET}" if (n_va_img == n_va_lbl and n_va_img > 0) else f"{RED}Mismatch ({n_va_img} imgs vs {n_va_lbl} lbls){RESET}"

    print(f"    • Train Split: {tr_match}")
    print(f"    • Val Split:   {va_match}")

    # Inspect data.yaml
    try:
        import yaml
        with open(data_yaml) as yf:
            ymeta = yaml.safe_load(yf)
        print(f"\n  {BOLD}Dataset Configuration (data.yaml):{RESET}")
        print(f"    • Target Class: {GREEN}{ymeta.get('names', {0: 'human'})}{RESET}")
        print(f"    • Train Path:   {ymeta.get('train')}")
        print(f"    • Val Path:     {ymeta.get('val')}")
    except Exception:
        pass

    # Quick sample annotation scan for class id confirmation
    sample_lbls = list(train_lbl.glob("*.txt"))[:100]
    total_boxes = 0
    empty_frames = 0
    class_ids = set()
    for lp in sample_lbls:
        text = lp.read_text().strip()
        if not text:
            empty_frames += 1
            continue
        for line in text.splitlines():
            parts = line.split()
            if parts:
                class_ids.add(parts[0])
                total_boxes += 1

    print(f"\n  {BOLD}Annotation Quality Sample (first {len(sample_lbls)} frames):{RESET}")
    print(f"    • Human Bounding Boxes: {CYAN}{total_boxes}{RESET}")
    print(f"    • Negative Frames (no humans): {empty_frames} ({empty_frames/len(sample_lbls)*100:.0f}%)")
    print(f"    • Detected Class IDs: {class_ids} (0 = single class 'human')")

    print(f"{BOLD}{WHITE}╰────────────────────────────────────────────────────────────────────────╯{RESET}")
    input(f"\n{DIM}Press Enter to return...{RESET}")


# --------------------------------------------------------------------------- #
# Main Application Loop
# --------------------------------------------------------------------------- #

def main():
    options = [
        {"label": "📥 Download Dataset", "desc": "Step 2: VisDrone2019-DET raw dataset (~2.5 GB)", "val": "0"},
        {"label": "📁 Prepare Dataset", "desc": "Step 3: VisDrone raw -> YOLO single-class format", "val": "1"},
        {"label": "🏋️ Train Model", "desc": "Step 4: YOLO26n / YOLO11n fine-tuning, resume, smoke test", "val": "2"},
        {"label": "🏆 Model Tournament", "desc": "Step 4.5: Overnight sequential model tournament (RTX 4050)", "val": "T"},
        {"label": "📦 Export & Quantize", "desc": "Step 5: Convert to NCNN / ONNX (fp16 / int8) for RPi 5", "val": "3"},
        {"label": "🎁 Package Deliverables", "desc": "Step 5.5: Benchmark, export, and create training-results.zip", "val": "P"},
        {"label": "📊 Hardware Benchmark", "desc": "Simulate RPi 5 4-thread CPU throughput & accuracy", "val": "4"},
        {"label": "🎥 Video / Live Inference", "desc": "Real-time bounding box detection & FPS monitoring", "val": "5"},
        {"label": "📈 Inspect Training Metrics", "desc": "View mAP curves, loss progression & best epochs", "val": "6"},
        {"label": "🩺 Dataset Health Check", "desc": "Verify split image/label pairing and annotations", "val": "7"},
        {"label": "🛠️ Install / Repair Env", "desc": "Step 1: Install or repair venv requirements.txt", "val": "I"},
    ]

    while True:
        choice = prompt_select_menu("Main Menu — Select Rescue Swarm Workflow", options)

        if choice == "0":
            menu_download_dataset()
        elif choice == "1":
            menu_prepare_dataset()
        elif choice == "2":
            menu_train()
        elif choice == "3":
            menu_export()
        elif choice == "P":
            menu_package_results()
        elif choice == "4":
            menu_evaluate()
        elif choice == "5":
            menu_inference()
        elif choice == "6":
            menu_inspect_metrics()
        elif choice == "7":
            menu_dataset_health()
        elif choice == "I":
            menu_install_dependencies()
        elif choice == "T":
            run_command([get_python_exe(), "training/tournament_runner.py", "--tui"], "Overnight Model Tournament")
        elif choice is None:
            print(f"\n{GREEN}Exiting Rescue Swarm Pipeline TUI. Fly safe!{RESET}\n")
            break


if __name__ == "__main__":
    main()
