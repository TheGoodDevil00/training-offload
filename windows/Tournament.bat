@echo off
title Overnight Drone-Footage Model Tournament (keep window open)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\scripts\tournament.ps1" %*
set RC=%ERRORLEVEL%
echo.
(if "%RC%"=="0" (echo RESULT: SUCCESS) else (echo RESULT: FAILED - read the messages above))
pause
exit /b %RC%
