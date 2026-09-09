"""
platform_utils.py
=================
Windows & Linux platform compatibility utilities:
  - Windows Console ANSI Virtual Terminal Processing (kernel32.SetConsoleMode)
  - Non-blocking universal keyboard reader (msvcrt on Windows, termios/select on Linux)
  - Sleep & Suspend Inhibitor (kernel32.SetThreadExecutionState for overnight runs)
  - Cross-platform Python executable detector (.venv / venv / sys.executable)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------- #
# Python Environment Resolver
# --------------------------------------------------------------------------- #

def get_project_python_exe(project_root: Optional[Path] = None) -> str:
    """
    Detects the virtual environment Python executable, prioritizing:
      1. .venv/Scripts/python.exe (Windows) or .venv/bin/python (Linux)
      2. venv/Scripts/python.exe (Windows) or venv/bin/python (Linux)
      3. sys.executable
    """
    if project_root is None:
        project_root = Path.cwd()

    candidates = []
    if os.name == "nt":
        candidates.append(project_root / ".venv" / "Scripts" / "python.exe")
        candidates.append(project_root / "venv" / "Scripts" / "python.exe")
    else:
        candidates.append(project_root / ".venv" / "bin" / "python")
        candidates.append(project_root / "venv" / "bin" / "python")

    for cand in candidates:
        if cand.exists():
            return str(cand)

    return sys.executable


# --------------------------------------------------------------------------- #
# Windows ANSI Virtual Terminal Mode
# --------------------------------------------------------------------------- #

def enable_windows_virtual_terminal():
    """
    Enables ENABLE_VIRTUAL_TERMINAL_PROCESSING on Windows console handles.
    Allows standard ANSI escape sequences (colors, cursor positioning, alternate screen)
    to render correctly in cmd.exe and powershell.exe without garbled characters.
    """
    if os.name != "nt":
        return

    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        STD_OUTPUT_HANDLE = -11
        STD_ERROR_HANDLE = -12

        for handle_id in (STD_OUTPUT_HANDLE, STD_ERROR_HANDLE):
            h = kernel32.GetStdHandle(handle_id)
            if h and h != -1:
                mode = wintypes.DWORD()
                if kernel32.GetConsoleMode(h, ctypes.byref(mode)):
                    # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
                    ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
                    ENABLE_PROCESSED_OUTPUT = 0x0001
                    kernel32.SetConsoleMode(
                        h, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING | ENABLE_PROCESSED_OUTPUT
                    )
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Windows Sleep & Suspend Inhibitor (Overnight Tournament)
# --------------------------------------------------------------------------- #

class WindowsSleepInhibitor:
    """
    Context manager that prevents Windows from sleeping, turning off the display,
    or entering standby during overnight training runs.
    Uses Win32 SetThreadExecutionState.
    """
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_AWAYMODE_REQUIRED = 0x00000040

    def __init__(self, active: bool = True):
        self.active = active and (os.name == "nt")

    def __enter__(self):
        if self.active:
            try:
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(
                    self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED | self.ES_AWAYMODE_REQUIRED
                )
            except Exception:
                pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.active:
            try:
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# Universal Non-blocking Keyboard Input Engine (Windows & Linux)
# --------------------------------------------------------------------------- #

def read_key_universal(timeout_s: float = 0.05) -> str:
    """
    Reads a single keypress or navigation event without pressing Enter.
    - On Windows: Uses msvcrt.kbhit() and msvcrt.getwch() / getch()
    - On Linux: Uses select on sys.stdin with termios raw mode
    Returns normalized strings: "UP", "DOWN", "LEFT", "RIGHT", "ENTER",
    "ESC", "BACKSPACE", "HOME", "END", or raw character ('1', 'q', etc.).
    """
    # 1. Non-interactive fallback
    if not sys.stdin.isatty():
        return ""

    # 2. Windows Implementation (msvcrt)
    if os.name == "nt":
        try:
            import msvcrt
            import time

            # Poll for input until timeout
            t0 = time.perf_counter()
            while (time.perf_counter() - t0) < timeout_s:
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    # Special prefix key in Windows console (0x00 or 0xe0)
                    if ch in ("\x00", "\xe0"):
                        ch2 = msvcrt.getwch()
                        # Extended scan codes
                        scan_map = {
                            "H": "UP",
                            "P": "DOWN",
                            "K": "LEFT",
                            "M": "RIGHT",
                            "G": "HOME",
                            "O": "END",
                            "I": "PAGE_UP",
                            "Q": "PAGE_DOWN",
                            "S": "DELETE",
                        }
                        return scan_map.get(ch2, "")
                    elif ch in ("\r", "\n"):
                        return "ENTER"
                    elif ch in ("\x08", "\x7f"):
                        return "BACKSPACE"
                    elif ch == "\x1b":
                        return "ESC"
                    elif ch == "\x03":  # Ctrl+C
                        raise KeyboardInterrupt
                    return ch
                time.sleep(0.01)
            return ""
        except ImportError:
            return ""

    # 3. Linux / POSIX Implementation (select + termios)
    try:
        import select
        fd = sys.stdin.fileno()
        r, _, _ = select.select([fd], [], [], timeout_s)
        if not r:
            return ""

        raw_byte = os.read(fd, 1)
        if not raw_byte:
            return ""

        ch = raw_byte.decode("utf-8", errors="ignore")
        if ch == "\x1b":
            # Read escape sequence
            seq = ""
            while True:
                r_seq, _, _ = select.select([fd], [], [], 0.02)
                if not r_seq:
                    break
                chunk = os.read(fd, 32).decode("utf-8", errors="ignore")
                if not chunk:
                    break
                seq += chunk

            # Decode sequence
            if seq in ("[A", "OA") or seq.endswith("A") or "57416" in seq:
                return "UP"
            if seq in ("[B", "OB") or seq.endswith("B") or "57417" in seq:
                return "DOWN"
            if seq in ("[C", "OC") or seq.endswith("C") or "57418" in seq:
                return "RIGHT"
            if seq in ("[D", "OD") or seq.endswith("D") or "57419" in seq:
                return "LEFT"
            if "H" in seq or "1~" in seq:
                return "HOME"
            if "F" in seq or "4~" in seq:
                return "END"
            if "5~" in seq:
                return "PAGE_UP"
            if "6~" in seq:
                return "PAGE_DOWN"
            return "ESC"
        elif ch in ("\r", "\n"):
            return "ENTER"
        elif ch in ("\x7f", "\x08"):
            return "BACKSPACE"
        elif ch == "\x03":
            raise KeyboardInterrupt
        return ch
    except Exception:
        return ""
