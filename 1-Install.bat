@echo off
title STEP 1/5 - Install (one-time setup)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\1_setup.ps1" %*
set RC=%ERRORLEVEL%
echo.
(if "%RC%"=="0" (echo RESULT: SUCCESS) else (echo RESULT: FAILED - read the messages above))
pause
exit /b %RC%
