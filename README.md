# 🚀 High-Quality Batch Background Remover (GPU / CPU)

An AI-powered batch background remover designed specifically to extract and process images directly from **ZIP archives**, maintaining **original image resolutions**, **preserving original filenames**, and unleashing **full NVIDIA GPU power (RTX 4050)** with automatic **CPU fallback**.

---

## ✨ Features

- **Direct ZIP In & Out**: Processes all images inside a `.zip` archive without manual extraction, and packages the result into a clean output `.zip` file.
- **Strict Filename & Structure Preservation**: Retains the exact names and directory hierarchies of your files (e.g. `photos/product_01.jpg` -> `photos/product_01.png`).
- **Highest Quality Transparent PNGs**: Uses state-of-the-art **BiRefNet** (Bilateral Reference Network) and **RMBG** models for razor-sharp edge detection and hair strands preservation.
- **Full NVIDIA GPU Power (CUDA)**: Utilizes NVIDIA Tensor Cores with FP16 half precision for maximum speed on your **NVIDIA GeForce RTX 4050**.
- **Automatic CPU Fallback**: Automatically switches to CPU if GPU is unavailable or if requested.
- **Dual Interfaces**:
  - **Modern Web GUI**: Drag-and-drop upload, live GPU hardware monitor, cutout confidence sliders, and 1-click download.
  - **Command-Line Interface (CLI)**: High-speed batch processing for scripts or terminal workflows.

---

## 🚀 Quick Start (Windows)

### 1. One-Click Setup
Double-click `install.bat` or run:
```powershell
.\install.bat
```
This automatically configures the Python 3.11 virtual environment and installs PyTorch with CUDA 12.4 support.

### 2. Launch the Web Interface
Double-click `start_gui.bat` or run:
```powershell
.\venv\Scripts\python.exe app.py
```
Open your browser to: **http://127.0.0.1:7860**

---

## 💻 Command Line Interface (CLI)

You can process any ZIP archive directly from the command line:

```powershell
.\venv\Scripts\python.exe cli.py --input "C:\path\to\my_images.zip"
```

### Advanced CLI Options:
```text
options:
  -i, --input PATH          Path to input ZIP archive containing images (Required)
  -o, --output PATH         Custom path for output ZIP (Default: <input>_nobg.zip)
  -e, --extract FOLDER      Also extract the processed PNG images to a folder
  -m, --model MODEL         Model architecture:
                              birefnet-general  (Default, Ultra Quality SOTA)
                              birefnet-portrait (Best for people / portraits)
                              rmbg-1.4          (Fast high-precision)
                              isnet-rembg       (ISNet rembg engine)
  -d, --device DEVICE       Compute device: 'auto' (Default, detects GPU), 'cuda', or 'cpu'
  -t, --threshold FLOAT     Cutout confidence threshold 0.1 to 0.9 (Default: 0.5)
  -f, --feather INT         Edge feathering radius 0 to 5 (Default: 0 for sharp)
  --no-fp16                 Disable FP16 half precision on CUDA
```

### Examples:
- Process using full NVIDIA GPU:
  ```powershell
  .\venv\Scripts\python.exe cli.py -i images.zip -d cuda -m birefnet-general
  ```
- Process using CPU:
  ```powershell
  .\venv\Scripts\python.exe cli.py -i images.zip -d cpu
  ```
- Process and extract directly to a folder:
  ```powershell
  .\venv\Scripts\python.exe cli.py -i images.zip -e "C:\MyOutputFolder"
  ```

---

## 🧠 Supported Models

| Model | Best For | Speed | Quality |
|---|---|---|---|
| **BiRefNet-General** *(Default)* | General objects, e-commerce, complex edges | Fast (GPU) | 🌟 Ultra (SOTA) |
| **BiRefNet-Portrait** | Portraits, hair, people | Fast (GPU) | 🌟 Ultra (SOTA) |
| **RMBG-1.4** | Clean products, general cutouts | Very Fast | ⭐ High |
| **ISNet-rembg** | Classic cutouts, lightweight | Fast | ⭐ Good |
