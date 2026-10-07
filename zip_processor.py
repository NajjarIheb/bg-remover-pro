"""
ZIP Archive Processor for Batch Background Removal.
Extracts images from ZIP, processes each through the background engine,
and packages transparent PNGs into a new ZIP archive while strictly preserving
original filenames and directory hierarchy.
"""

import os
import io
import zipfile
import logging
from typing import Callable, Optional, List, Tuple
from PIL import Image
from engine import BackgroundRemovalEngine

logger = logging.getLogger("zip_processor")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

class ZipBatchProcessor:
    def __init__(self, engine: BackgroundRemovalEngine):
        self.engine = engine

    def is_image_file(self, filename: str) -> bool:
        ext = os.path.splitext(filename)[1].lower()
        return ext in SUPPORTED_EXTENSIONS and not os.path.basename(filename).startswith("._")

    def inspect_zip(self, zip_path: str) -> List[str]:
        """Returns a list of image filenames found in the ZIP archive."""
        image_files = []
        with zipfile.ZipFile(zip_path, 'r') as archive:
            for item in archive.infolist():
                if not item.is_dir() and self.is_image_file(item.filename):
                    image_files.append(item.filename)
        return image_files

    def process_zip(
        self,
        input_zip_path: str,
        output_zip_path: str,
        extract_folder: Optional[str] = None,
        threshold: float = 0.5,
        feather_radius: int = 0,
        progress_callback: Optional[Callable[[int, int, str, Optional[Image.Image]], None]] = None
    ) -> Tuple[int, int, List[str]]:
        """
        Process all images inside a ZIP archive.
        
        Args:
            input_zip_path: Path to input ZIP archive.
            output_zip_path: Path to destination ZIP file.
            extract_folder: Optional directory to also save extracted images.
            threshold: Alpha cutoff threshold.
            feather_radius: Edge feathering radius.
            progress_callback: Callback(index, total, filename, processed_pil_image).
            
        Returns:
            Tuple of (successful_count, total_count, error_list)
        """
        if not os.path.exists(input_zip_path):
            raise FileNotFoundError(f"Input ZIP not found: {input_zip_path}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_zip_path)), exist_ok=True)
        if extract_folder:
            os.makedirs(extract_folder, exist_ok=True)

        image_files = self.inspect_zip(input_zip_path)
        total_images = len(image_files)
        logger.info(f"Found {total_images} image(s) in {input_zip_path}")

        if total_images == 0:
            raise ValueError(f"No valid image files found in {input_zip_path}. Supported: {SUPPORTED_EXTENSIONS}")

        successful = 0
        errors = []

        with zipfile.ZipFile(input_zip_path, 'r') as in_zip, \
             zipfile.ZipFile(output_zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as out_zip:

            for idx, file_rel_path in enumerate(image_files, start=1):
                try:
                    # Read image from ZIP into memory
                    image_bytes = in_zip.read(file_rel_path)
                    image_stream = io.BytesIO(image_bytes)
                    input_img = Image.open(image_stream)

                    # Remove background with highest quality
                    output_img = self.engine.process_image(
                        input_img,
                        threshold=threshold,
                        feather_radius=feather_radius
                    )

                    # Maintain identical name, switching extension to .png to store alpha channel losslessly
                    base_dir, base_file = os.path.split(file_rel_path)
                    base_name, _ = os.path.splitext(base_file)
                    out_filename = f"{base_name}.png"
                    out_rel_path = os.path.join(base_dir, out_filename).replace("\\", "/")

                    # Encode to PNG in memory
                    out_buffer = io.BytesIO()
                    output_img.save(out_buffer, format="PNG", optimize=False)
                    out_bytes = out_buffer.getvalue()

                    # Write to output ZIP
                    out_zip.writestr(out_rel_path, out_bytes)

                    # Optionally save to extracted folder
                    if extract_folder:
                        dest_file_path = os.path.join(extract_folder, out_rel_path)
                        os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
                        with open(dest_file_path, "wb") as f:
                            f.write(out_bytes)

                    successful += 1

                    if progress_callback:
                        progress_callback(idx, total_images, file_rel_path, output_img)

                except Exception as e:
                    err_msg = f"Failed processing {file_rel_path}: {e}"
                    logger.error(err_msg)
                    errors.append(err_msg)
                    if progress_callback:
                        progress_callback(idx, total_images, file_rel_path, None)

        logger.info(f"Finished processing ZIP: {successful}/{total_images} images successful.")
        return successful, total_images, errors
