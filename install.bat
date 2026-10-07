@echo off
title High-Quality Batch Background Remover - Setup
echo ========================================================
echo   Setting up High-Quality Batch Background Remover
echo ========================================================
echo.

if not exist venv (
    echo [*] Creating Python 3.11 virtual environment...
    py -3.11 -m venv venv
    if errorlevel 1 (
        echo [!] Failed to create venv using py -3.11. Trying python...
        python -m venv venv
    )
)

echo [*] Upgrading pip...
venv\Scripts\python.exe -m pip install --upgrade pip

echo [*] Installing PyTorch with NVIDIA CUDA 12.4 support...
venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo [*] Installing AI models and dependencies...
venv\Scripts\python.exe -m pip install transformers timm accelerate pillow opencv-python-headless tqdm gradio rembg psutil einops

echo.
echo ========================================================
echo [*] Testing CUDA GPU acceleration...
venv\Scripts\python.exe -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
echo ========================================================
echo Setup complete! Double-click 'start_gui.bat' to launch.
pause
