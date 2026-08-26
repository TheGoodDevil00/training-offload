@echo off
title STEP 4/5 - Train (keep window open)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\4_train.ps1" %*
set RC=%ERRORLEVEL%
echo.
(if "%RC%"=="0" (echo RESULT: SUCCESS) else (echo RESULT: FAILED - read the messages above))
pause
exit /b %RC%
