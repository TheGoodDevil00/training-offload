@echo off
title Rescue Swarm Human Detector — Unified Pipeline TUI
"%~dp0..\.venv\Scripts\python.exe" "%~dp0..\training\tui.py" %*
set RC=%ERRORLEVEL%
if not "%RC%"=="0" (
    echo.
    echo RESULT: FAILED - exit code %RC%
    pause
)
exit /b %RC%
