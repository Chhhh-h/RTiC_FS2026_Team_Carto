#!/usr/bin/env python3
"""Convert multi-band one-hot TIFF masks into single-channel class-index masks.

Many MMSegmentation/SegFormer configs assume the annotation (gt_semantic_seg)
file is a *single channel* image where each pixel value is the class id:
  0..(num_classes-1), with 255 commonly used as ignore.

If your annotations are stored as a 7-band TIFF where band k is a {0,1} mask for
class k, this script converts them into a 2D mask.

Edit the constants in the "CONFIG" section below, then run:
    python scripts/convert_onehot_tif_to_mask.py

Notes:
- Assumes classes are mutually exclusive per pixel (single-label semantic seg).
- If a pixel has no positive class (all bands ~0), you can map it to background
  (0), or ignore_index (255).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# -----------------------------
# CONFIG (edit these)
# -----------------------------
# Input can be a folder or a single .tif file
INPUT: Path = Path("/cluster/scratch/pangyi/reto/data/train/train")

# Output folder (or a .tif file path if INPUT is a file)
OUTPUT: Path = Path("/cluster/scratch/pangyi/reto/data/train/train_mask")

# Pattern used when INPUT is a folder
PATTERN: str = "*.tif"

# Recurse into subfolders when INPUT is a folder
RECURSIVE: bool = True

# Number of bands/classes in the one-hot mask
NUM_CLASSES: int = 7

# What value to assign when a pixel has no class (all bands==0)
# Common choices: 255(ignore) or 0(background)
ZERO_SUM_TO: int = 255

# Output dtype: 'uint8' is fine for <=255 classes; use 'uint16' otherwise
OUTPUT_DTYPE: str = "uint8"  # 'uint8' or 'uint16'

# Write output TIFF without compression (i.e. not LZW)
WRITE_COMPRESSION = None


@dataclass(frozen=True)
class ConvertConfig:
    num_classes: int
    zero_sum_to: int = 255  # 255=ignore by default
    output_dtype: str = "uint8"  # 'uint8' or 'uint16'


def _iter_inputs(input_path: Path, pattern: str, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(str(input_path))

    globber = input_path.rglob if recursive else input_path.glob
    files = [p for p in globber(pattern) if p.is_file()]
    files.sort()
    return files


def _ensure_onehot_shape(arr: np.ndarray, num_classes: int) -> tuple[np.ndarray, int]:
    """Return (arr, band_axis) where arr has shape (..., num_classes, ...)."""
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array for one-hot mask, got shape={arr.shape}")

    # Common layouts: (H, W, C) or (C, H, W)
    if arr.shape[2] == num_classes:
        return arr, 2
    if arr.shape[0] == num_classes:
        return arr, 0
    if arr.shape[1] == num_classes:
        return arr, 1

    raise ValueError(
        f"Cannot find a band axis of size {num_classes} in shape={arr.shape}"
    )


def convert_onehot_array_to_mask(arr: np.ndarray, cfg: ConvertConfig) -> np.ndarray:
    """Convert a one-hot (3D) array into a 2D class-index mask."""
    arr = np.asarray(arr)

    arr, band_axis = _ensure_onehot_shape(arr, cfg.num_classes)

    # Move band axis to last for easier handling: (H, W, C)
    if band_axis != 2:
        arr = np.moveaxis(arr, band_axis, 2)

    # Work in float for robustness
    arr_f = arr.astype(np.float32, copy=False)

    # For true one-hot {0,1} masks: a pixel is labeled iff sum(bands) > 0
    # (no thresholding needed)
    sum_v = np.sum(arr_f, axis=2)
    has_class = sum_v > 0

    # Argmax gives class index
    mask = np.argmax(arr_f, axis=2).astype(np.int32)

    # Handle empty pixels (no class)
    if cfg.zero_sum_to is not None:
        mask = mask.astype(np.int32, copy=False)
        mask[~has_class] = int(cfg.zero_sum_to)

    if cfg.output_dtype == "uint8":
        return mask.astype(np.uint8, copy=False)
    if cfg.output_dtype == "uint16":
        return mask.astype(np.uint16, copy=False)

    raise ValueError(f"Unsupported output_dtype={cfg.output_dtype}")


def convert_file(in_path: Path, out_path: Path, cfg: ConvertConfig) -> None:
    try:
        import tifffile  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError("Missing dependency `tifffile`. Install: pip install tifffile") from e

    arr = tifffile.imread(str(in_path))
    mask = convert_onehot_array_to_mask(arr, cfg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out_path), mask, compression=WRITE_COMPRESSION)


def _relpath_or_name(path: Path, base: Path) -> Path:
    try:
        return path.relative_to(base)
    except Exception:
        return Path(path.name)


def main() -> int:
    cfg = ConvertConfig(
        num_classes=NUM_CLASSES,
        zero_sum_to=ZERO_SUM_TO,
        output_dtype=OUTPUT_DTYPE,
    )

    in_paths = _iter_inputs(INPUT, PATTERN, RECURSIVE)
    if not in_paths:
        raise SystemExit(f"No inputs matched: {INPUT} pattern={PATTERN}")

    # If input is a file, output can be a file path.
    if INPUT.is_file():
        out_path = OUTPUT
        if OUTPUT.is_dir():
            out_path = OUTPUT / INPUT.name
        convert_file(INPUT, out_path, cfg)
        print(f"wrote\t{out_path}")
        return 0

    # Folder mode: mirror relative structure under output
    for in_path in in_paths:
        rel = _relpath_or_name(in_path, INPUT)
        out_path = OUTPUT / rel
        convert_file(in_path, out_path, cfg)
        print(f"wrote\t{out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
