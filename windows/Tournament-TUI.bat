@echo off
title Overnight Drone Model Tournament — Workstation TUI
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\scripts\tournament.ps1" -TUI %*
set RC=%ERRORLEVEL%
if not "%RC%"=="0" (
    echo.
    echo RESULT: FAILED - read the messages above
    pause
)
exit /b %RC%
