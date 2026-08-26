@echo off
title QUICK TEST - verify pipeline end to end
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\scripts\4_train.ps1" -SmokeTest %*
set RC=%ERRORLEVEL%
echo.
(if "%RC%"=="0" (echo RESULT: SUCCESS) else (echo RESULT: FAILED - read the messages above))
pause
exit /b %RC%
