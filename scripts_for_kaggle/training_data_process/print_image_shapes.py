#!/usr/bin/env python3
"""Print image sizes (H, W, bands) for PNG/TIF files in a folder.

Edit the constants in the "CONFIG" section below to point to your folder.

Output columns:
    path\theight\twidth\tbands
"""

from __future__ import annotations

import warnings
from pathlib import Path

# -----------------------------
# CONFIG (edit these)
# -----------------------------
# Folder to scan. Use an absolute path or a path relative to where you run it.
FOLDER = Path("/cluster/scratch/pangyi/reto/data/origin/test_onechannel")

# Recursively scan subfolders.
RECURSIVE = True

# File patterns to include.
PATTERNS = ["*.png", "*.PNG", "*.tif", "*.TIF", "*.tiff", "*.TIFF"]

# Max number of files to print (0 = no limit).
LIMIT = 0

# Pillow safety limit: large images may trigger DecompressionBombWarning.
# If you trust the images (local dataset), you can silence/disable this.
ALLOW_LARGE_IMAGES = True
SILENCE_DECOMPRESSION_BOMB_WARNING = True

# TIFF note:
# Some TIFFs (e.g. multi-spectral with 7+ bands) cannot be decoded by Pillow.
# For such files, use `tifffile` to read metadata (shape/axes) instead.
PREFER_TIFFFILE_FOR_TIFF = True

# Exit code policy: when True, returns non-zero if any file fails.
RETURN_NONZERO_ON_ERROR = True


def _read_tiff_hw_bands_tifffile(path: Path) -> tuple[int, int, int]:
    """Read (H, W, bands) from a TIFF using `tifffile` metadata.

    This avoids Pillow's decoding limitations for multi-band TIFF.
    """
    try:
        import tifffile  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "tifffile is required to inspect some .tif/.tiff files. "
            "Install it via: pip install tifffile"
        ) from e

    with tifffile.TiffFile(path) as tif:
        # Prefer series metadata (most consistent across variants)
        series = tif.series[0]
        shape = series.shape
        axes = series.axes  # e.g. 'YXS', 'SYX', 'CYX'

        if not axes or len(shape) != len(axes):
            # Fallback to first page
            page = tif.pages[0]
            shape = page.shape
            # page.shape usually is (Y, X) or (Y, X, S)
            if len(shape) == 2:
                h, w = int(shape[0]), int(shape[1])
                return h, w, 1
            if len(shape) == 3:
                h, w, s = int(shape[0]), int(shape[1]), int(shape[2])
                return h, w, s
            raise RuntimeError(f"Unsupported TIFF shape: {shape}")

        # Map axis letters to indices
        axis_to_index = {ax: i for i, ax in enumerate(axes)}
        if 'Y' not in axis_to_index or 'X' not in axis_to_index:
            raise RuntimeError(f"Unexpected TIFF axes: {axes} shape={shape}")

        h = int(shape[axis_to_index['Y']])
        w = int(shape[axis_to_index['X']])

        # Bands can appear as 'S' (samples), 'C' (channels), or sometimes 'Z'
        for band_axis in ('S', 'C'):
            if band_axis in axis_to_index:
                b = int(shape[axis_to_index[band_axis]])
                return h, w, b

        # No explicit band axis -> single band
        return h, w, 1


def _num_bands_from_pil(image) -> int:
    # PIL: getbands() returns a tuple like ('R','G','B')
    try:
        bands = image.getbands()
        if bands:
            return len(bands)
    except Exception:
        pass

    # Fallbacks
    for attr in ("n_frames", "layers"):
        if hasattr(image, attr):
            try:
                value = int(getattr(image, attr))
                if value > 0:
                    return value
            except Exception:
                pass

    return 1


def read_image_hw_bands(path: Path) -> tuple[int, int, int]:
    """Return (height, width, bands) for an image file."""
    suffix = path.suffix.lower()
    if PREFER_TIFFFILE_FOR_TIFF and suffix in {'.tif', '.tiff'}:
        return _read_tiff_hw_bands_tifffile(path)

    # Prefer PIL (already used by MMSeg and works for png in most cases).
    try:
        from PIL import Image  # type: ignore

        if ALLOW_LARGE_IMAGES:
            # Disable the pixel limit check for this process.
            Image.MAX_IMAGE_PIXELS = None
        if SILENCE_DECOMPRESSION_BOMB_WARNING:
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)

        with Image.open(path) as img:
            width, height = img.size
            bands = _num_bands_from_pil(img)
            return height, width, bands
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Pillow (PIL) is not installed. Install it via: pip install pillow"
        ) from e
    except Exception:
        # If Pillow fails on TIFF, fall back to tifffile metadata.
        if suffix in {'.tif', '.tiff'}:
            return _read_tiff_hw_bands_tifffile(path)
        raise


def iter_files(folder: Path, recursive: bool, patterns: list[str]) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(str(folder))
    if not folder.is_dir():
        raise NotADirectoryError(str(folder))

    results: list[Path] = []
    for pattern in patterns:
        globber = folder.rglob if recursive else folder.glob
        results.extend([p for p in globber(pattern) if p.is_file()])

    # De-dup while keeping a stable order
    unique: dict[Path, None] = {}
    for p in sorted(results):
        unique[p] = None
    return list(unique.keys())


def main() -> int:
    files = iter_files(FOLDER, RECURSIVE, PATTERNS)
    if LIMIT and LIMIT > 0:
        files = files[:LIMIT]

    print("path\theight\twidth\tbands")

    bad = 0
    for path in files:
        try:
            h, w, b = read_image_hw_bands(path)
            print(f"{path}\t{h}\t{w}\t{b}")
        except Exception as e:
            bad += 1
            print(f"{path}\tERROR\t{e}")

    if bad:
        # non-zero to make it easy to detect issues in scripts
        return 2 if RETURN_NONZERO_ON_ERROR else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
