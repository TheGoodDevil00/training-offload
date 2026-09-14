@echo off
title Rescue Swarm Human Detector — Unified Pipeline TUI

if not exist "%~dp0..\.venv\Scripts\python.exe" (
    echo.
    echo ====================================================================
    echo   [!] Virtual environment (.venv) was not found.
    echo   [*] Running Step 1 Setup now to install Python and dependencies...
    echo ====================================================================
    echo.
    call "%~dp01-Install.bat"
    if not exist "%~dp0..\.venv\Scripts\python.exe" (
        echo.
        echo [ERROR] Setup did not complete successfully.
        pause
        exit /b 1
    )
)

"%~dp0..\.venv\Scripts\python.exe" "%~dp0..\training\tui.py" %*
set RC=%ERRORLEVEL%
if not "%RC%"=="0" (
    echo.
    echo RESULT: FAILED - exit code %RC%
    pause
)
exit /b %RC%
