"""
High-Performance Background Removal Engine.
Supports SOTA AI models (ISNet, BiRefNet, U2Net) with full NVIDIA GPU power
(DirectML / CUDA) and seamless CPU fallback.
"""

import os
import gc
import logging
import subprocess
from typing import Optional, Tuple, Union, Callable, List
from PIL import Image, ImageOps, ImageFilter
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bg_engine")

def get_nvidia_device_id() -> int:
    """Find the DirectML device_id corresponding to NVIDIA GPU."""
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False
        )
        if res.returncode == 0:
            lines = [l.strip() for l in res.stdout.strip().splitlines() if l.strip()]
            for idx, name in enumerate(lines):
                if any(k in name.lower() for k in ["nvidia", "geforce", "rtx", "gtx"]):
                    return idx
    except Exception:
        pass
    return 1

def detect_gpu_hardware() -> Tuple[bool, str, str]:
    """Detect NVIDIA GPU hardware using nvidia-smi, torch, or system info."""
    try:
        import torch
        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0), "CUDA"
    except Exception:
        pass

    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False
        )
        if res.returncode == 0 and res.stdout.strip():
            gpu_name = res.stdout.strip().split(",")[0]
            return True, gpu_name, "NVIDIA-SMI"
    except Exception:
        pass

    return False, "CPU", "None"

def get_live_gpu_stats() -> dict:
    """Returns real-time GPU statistics from nvidia-smi."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = [x.strip() for x in res.stdout.strip().split(",")]
            if len(parts) >= 6:
                return {
                    "has_gpu": True,
                    "name": parts[0],
                    "utilization": f"{parts[1]}%",
                    "mem_used": f"{parts[2]} MB",
                    "mem_total": f"{parts[3]} MB",
                    "temp": f"{parts[4]}°C",
                    "power": f"{parts[5]}W"
                }
    except Exception:
        pass
    return {"has_gpu": False, "name": "CPU", "utilization": "N/A", "mem_used": "N/A", "mem_total": "N/A", "temp": "N/A", "power": "N/A"}

def get_model_cache_info(model_key: str) -> dict:
    """Check if model weights are already downloaded locally."""
    rembg_home = os.path.expanduser(os.path.join("~", ".rembg", "models"))
    model_files = {
        "isnet-general-use": os.path.join(rembg_home, "isnet-general-use", "isnet-general-use.onnx"),
        "isnet-rembg": os.path.join(rembg_home, "isnet-general-use", "isnet-general-use.onnx"),
        "birefnet-general": os.path.join(rembg_home, "birefnet-general", "birefnet-general.onnx"),
        "birefnet-portrait": os.path.join(rembg_home, "birefnet-portrait", "birefnet-portrait.onnx"),
        "u2net": os.path.join(rembg_home, "u2net", "u2net.onnx"),
    }
    path = model_files.get(model_key.lower())
    if path and os.path.exists(path) and os.path.getsize(path) > 1000000:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        return {"cached": True, "size_mb": round(size_mb, 1), "path": path}
    return {"cached": False, "size_mb": 0, "path": path}

class BackgroundRemovalEngine:
    def __init__(
        self,
        model_name: str = "isnet-general-use",
        device: str = "auto",
        fp16: bool = True
    ):
        """
        Initialize the Background Removal Engine.
        
        Args:
            model_name: Model key ('isnet-general-use', 'birefnet-general', 'u2net')
            device: 'auto', 'cuda', 'dml', or 'cpu'
            fp16: Use FP16 half precision when supported
        """
        self.model_name = model_name
        self.requested_device = device.lower()
        self.fp16 = fp16
        self.active_provider = "CPU"
        self.gpu_detected, self.gpu_name, _ = detect_gpu_hardware()
        self.session = None
        self._load_model()

    def get_hardware_info(self) -> dict:
        info = {
            "device": self.requested_device,
            "provider": self.active_provider,
            "gpu_detected": self.gpu_detected,
            "gpu_name": self.gpu_name,
            "device_name": self.gpu_name if "CPU" not in self.active_provider else "CPU"
        }
        live_stats = get_live_gpu_stats()
        if live_stats.get("has_gpu"):
            info["gpu_stats"] = live_stats
        return info

    def _determine_providers(self) -> list:
        import onnxruntime as ort
        available = ort.get_available_providers()
        logger.info(f"Available ONNX execution providers: {available}")

        if self.requested_device == "cpu":
            self.active_provider = "CPUExecutionProvider"
            return ["CPUExecutionProvider"]

        desired = []
        if "CUDAExecutionProvider" in available:
            desired.append("CUDAExecutionProvider")
            self.active_provider = f"CUDAExecutionProvider (GPU: {self.gpu_name})"
        elif "DmlExecutionProvider" in available:
            dev_id = get_nvidia_device_id()
            desired.append(("DmlExecutionProvider", {"device_id": dev_id}))
            self.active_provider = f"DirectML Dedicated GPU Adapter {dev_id} ({self.gpu_name})"
        
        desired.append("CPUExecutionProvider")
        if not self.active_provider or self.active_provider == "CPU":
            self.active_provider = "CPUExecutionProvider"

        logger.info(f"Using execution provider: {desired[0]}")
        return desired

    def _load_model(self):
        logger.info(f"Loading model '{self.model_name}' (device mode: {self.requested_device})...")
        import rembg

        providers = self._determine_providers()
        model_key = self.model_name.lower()
        if "birefnet-portrait" in model_key:
            session_model = "birefnet-portrait"
        elif "birefnet" in model_key:
            session_model = "birefnet-general"
        elif "u2net" in model_key:
            session_model = "u2net"
        else:
            session_model = "isnet-general-use"

        logger.info(f"Initializing rembg session '{session_model}' with providers: {providers}")
        self.session = rembg.new_session(session_model, providers=providers)
        logger.info(f"Engine successfully loaded '{session_model}' on {self.active_provider}")

    def process_image(
        self,
        image: Union[Image.Image, np.ndarray, str],
        threshold: float = 0.5,
        feather_radius: int = 0,
        background_color: Optional[Tuple[int, int, int]] = None
    ) -> Image.Image:
        """
        Removes background from an image while preserving full original quality and resolution.
        """
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Handle EXIF orientation
        image = ImageOps.exif_transpose(image)
        orig_w, orig_h = image.size
        orig_rgb = image.convert("RGB")

        if self.session is None:
            self._load_model()

        import rembg
        output = rembg.remove(
            orig_rgb,
            session=self.session,
            alpha_matting=False,
            post_process_mask=True
        )

        # Ensure correct size
        if output.size != (orig_w, orig_h):
            output = output.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

        # Optional thresholding & feathering
        if threshold != 0.5 or feather_radius > 0:
            r, g, b, alpha = output.split()
            if threshold != 0.5:
                lut = [255 if (val / 255.0) >= threshold else int(val * (1.0 / (threshold * 2))) for val in range(256)]
                alpha = alpha.point(lut)
            if feather_radius > 0:
                alpha = alpha.filter(ImageFilter.GaussianBlur(feather_radius))
            output = Image.merge("RGBA", (r, g, b, alpha))

        if background_color is not None:
            bg = Image.new("RGBA", (orig_w, orig_h), (*background_color, 255))
            bg.paste(output, (0, 0), output)
            return bg.convert("RGB")

        return output

    def switch_device(self, new_device: str):
        if new_device.lower() == self.requested_device:
            return
        self.requested_device = new_device.lower()
        self.session = None
        gc.collect()
        self._load_model()

    def switch_model(self, new_model_name: str):
        if new_model_name == self.model_name:
            return
        self.model_name = new_model_name
        self.session = None
        gc.collect()
        self._load_model()
