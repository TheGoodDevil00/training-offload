@echo off
title STEP 3/5 - Prepare dataset
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\3_prepare_dataset.ps1" %*
set RC=%ERRORLEVEL%
echo.
(if "%RC%"=="0" (echo RESULT: SUCCESS) else (echo RESULT: FAILED - read the messages above))
pause
exit /b %RC%
