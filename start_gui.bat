@echo off
title High-Quality Batch Background Remover - GUI
echo ========================================================
echo   Launching High-Quality Batch Background Remover GUI
echo ========================================================
echo.
if not exist venv\Scripts\python.exe (
    echo [!] Virtual environment not found. Running install.bat first...
    call install.bat
)

venv\Scripts\python.exe app.py
pause
