@echo off
title STEP 2/5 - Download dataset (~2.5 GB)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\2_download_data.ps1" %*
set RC=%ERRORLEVEL%
echo.
(if "%RC%"=="0" (echo RESULT: SUCCESS) else (echo RESULT: FAILED - read the messages above))
pause
exit /b %RC%
