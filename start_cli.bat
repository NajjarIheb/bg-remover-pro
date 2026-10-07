@echo off
title High-Quality Batch Background Remover - CLI
echo ========================================================
echo   High-Quality Batch Background Remover - Command Line
echo ========================================================
echo.

set /p INPUT_ZIP="Enter the path to your input ZIP file: "
if "%INPUT_ZIP%"=="" (
    echo [!] No file entered. Exiting.
    pause
    exit /b
)

venv\Scripts\python.exe cli.py --input %INPUT_ZIP% --device auto --model birefnet-general
pause
