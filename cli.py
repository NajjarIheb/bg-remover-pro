"""
Command Line Interface for High-Quality Batch Background Remover.
Run high-performance background removal with full NVIDIA GPU power or CPU fallback.
"""

import os
import sys
import time
import argparse

# Enable UTF-8 for Windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from tqdm import tqdm
from engine import BackgroundRemovalEngine
from zip_processor import ZipBatchProcessor

def format_bytes(size_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def print_banner(info: dict):
    print("=" * 65)
    print("        [+] HIGH-QUALITY BATCH BACKGROUND REMOVER [+]        ")
    print("=" * 65)
    print(f" Target Device     : {info.get('device_name')}")
    print(f" Active Engine     : {info.get('provider')}")
    if info.get("vram_total_mb"):
        print(f" Total VRAM        : {info.get('vram_total_mb')} MB")
        print(f" Active VRAM       : {info.get('vram_allocated_mb', 0)} MB")
    print("=" * 65)

def main():
    parser = argparse.ArgumentParser(
        description="High-Quality Batch Background Remover from ZIP archives using GPU/CPU."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input ZIP archive containing images."
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Path for output ZIP file (default: <input_name>_nobg.zip)."
    )
    parser.add_argument(
        "-e", "--extract",
        default=None,
        help="Optional destination folder to also extract processed images directly."
    )
    parser.add_argument(
        "-m", "--model",
        default="birefnet-general",
        choices=["birefnet-general", "birefnet-portrait", "rmbg-1.4", "isnet-rembg"],
        help="AI Model architecture (Default: birefnet-general - Ultra Quality)."
    )
    parser.add_argument(
        "-d", "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device: 'auto' (detects NVIDIA GPU), 'cuda', or 'cpu'."
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16 half precision on GPU (default is FP16 enabled for speed & memory)."
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.5,
        help="Alpha cutout threshold (0.0 to 1.0, default: 0.5)."
    )
    parser.add_argument(
        "-f", "--feather",
        type=int,
        default=0,
        help="Edge feathering/blur radius (0 for sharp edges, default: 0)."
    )

    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"[!] Error: Input file '{input_path}' does not exist.")
        sys.exit(1)

    if not args.output:
        base_stem = os.path.splitext(input_path)[0]
        output_path = f"{base_stem}_nobg.zip"
    else:
        output_path = os.path.abspath(args.output)

    print("\n[*] Initializing Background Removal Engine...")
    start_time = time.time()

    try:
        engine = BackgroundRemovalEngine(
            model_name=args.model,
            device=args.device,
            fp16=not args.no_fp16
        )
    except Exception as e:
        print(f"[!] Failed to initialize engine: {e}")
        sys.exit(1)

    hw_info = engine.get_hardware_info()
    print_banner(hw_info)

    processor = ZipBatchProcessor(engine)
    image_files = processor.inspect_zip(input_path)
    total_count = len(image_files)

    if total_count == 0:
        print("[!] No compatible images found in the ZIP archive.")
        sys.exit(1)

    print(f"[*] Input Archive  : {os.path.basename(input_path)} ({total_count} images)")
    print(f"[*] Output Archive : {output_path}")
    if args.extract:
        print(f"[*] Extract Folder : {args.extract}")
    print(f"[*] Selected Model : {args.model}")
    print("-" * 65)

    pbar = tqdm(total=total_count, desc="Processing images", unit="img")

    def on_progress(idx, total, filename, _img):
        pbar.update(1)
        pbar.set_postfix_str(f"Current: {os.path.basename(filename)[:25]}")

    success_cnt, total, errors = processor.process_zip(
        input_zip_path=input_path,
        output_zip_path=output_path,
        extract_folder=args.extract,
        threshold=args.threshold,
        feather_radius=args.feather,
        progress_callback=on_progress
    )
    pbar.close()

    elapsed = time.time() - start_time
    out_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0

    print("\n" + "=" * 65)
    print("           [OK] BATCH PROCESSING COMPLETE           ")
    print(f"[*] Successfully processed : {success_cnt}/{total} images")
    print(f"[*] Total time elapsed     : {elapsed:.2f} seconds ({elapsed / max(1, total):.2f}s per image)")
    print(f"[*] Output Archive Size    : {format_bytes(out_size)}")
    print(f"[*] Output File Saved To   : {output_path}")
    if errors:
        print(f"[!] Warnings / Errors ({len(errors)}):")
        for err in errors[:5]:
            print(f"   - {err}")
    print("=" * 65)

if __name__ == "__main__":
    main()
