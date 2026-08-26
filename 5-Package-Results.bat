@echo off
title STEP 5/5 - Package results
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\5_package_results.ps1" %*
set RC=%ERRORLEVEL%
echo.
(if "%RC%"=="0" (echo RESULT: SUCCESS) else (echo RESULT: FAILED - read the messages above))
pause
exit /b %RC%
