#!/usr/bin/env python3
"""Utilities to inspect TIFF band statistics.

This file provides a single function:

  print_tiff_stats(path)

It prints:
- TIFF series shape/axes/dtype
- Basic tag hints (compression/photometric when available)
- Per-band pixel statistics (min/max/mean/std + unique preview for integer data)

Notes:
- Multi-band TIFFs (e.g. 7 samples/pixel) often cannot be decoded by Pillow.
  This uses `tifffile` instead.
- For very large images, statistics are computed on a strided sample by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np


@dataclass(frozen=True)
class TiffStatsConfig:
    # Max number of pixels to use per band when computing summary stats.
    max_samples_per_band: int = 5_000_000
    # For integer masks, preview up to N unique values.
    max_unique_preview: int = 64
    # If there are more than this many unique values, don't print the full list.
    max_unique_to_list: int = 256


def _format_number(x: float) -> str:
    if np.isnan(x):
        return "nan"
    if np.isinf(x):
        return "inf" if x > 0 else "-inf"
    # Keep it readable but not too verbose
    if abs(x) >= 1e6 or (abs(x) > 0 and abs(x) < 1e-3):
        return f"{x:.6e}"
    return f"{x:.6f}".rstrip("0").rstrip(".")


def _sample_view(arr: np.ndarray, max_samples: int) -> tuple[np.ndarray, Optional[int]]:
    """Return (sample, step).

    If step is None, sample is the full array flattened.
    If step is an int, sample is a strided flattened view and step indicates stride.
    """
    flat = arr.reshape(-1)
    n = flat.size
    if n <= max_samples:
        return flat, None

    # Choose a stride so that n/step ~= max_samples
    step = int(np.ceil(n / max_samples))
    return flat[::step], step


def _infer_band_axis(axes: str) -> Optional[str]:
    # Common conventions:
    # - OME: 'TCZYX' etc
    # - tifffile uses 'YXS' for samples
    for cand in ("S", "C"):
        if cand in axes:
            return cand
    return None


def _band_name(band_axis: Optional[str], band_index: int) -> str:
    if band_axis is None:
        return "band"
    return f"{band_axis}{band_index}"


def _get_tags_summary(tif) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        page0 = tif.pages[0]
    except Exception:
        return out

    # Best-effort; tags vary by file
    for key in ("Compression", "PhotometricInterpretation", "BitsPerSample", "SamplesPerPixel"):
        try:
            tag = page0.tags.get(key)
            if tag is not None:
                out[key] = str(tag.value)
        except Exception:
            pass

    return out


def print_tiff_stats(path: Union[str, Path], config: TiffStatsConfig = TiffStatsConfig()) -> None:
    """Print TIFF metadata + per-band pixel statistics.

    Args:
        path: Path to a .tif/.tiff file.
        config: Sampling/printing configuration.
    """
    try:
        import tifffile  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Missing dependency `tifffile`. Install it via: pip install tifffile"
        ) from e

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = getattr(series, "axes", "") or ""
        shape = tuple(int(x) for x in series.shape)
        dtype = np.dtype(series.dtype)

        tags = _get_tags_summary(tif)

        print(f"path\t{path}")
        print(f"series_shape\t{shape}")
        print(f"axes\t{axes if axes else '(unknown)'}")
        print(f"dtype\t{dtype}")
        if tags:
            for k, v in tags.items():
                print(f"tag:{k}\t{v}")

        # `tifffile.memmap` avoids reading the entire image into RAM, but it
        # only works for certain TIFF layouts (typically uncompressed and
        # contiguous). Many real-world TIFFs (compressed/tiled) are not
        # memory-mappable.
        try:
            arr = tifffile.memmap(str(path))
        except Exception as e:
            print(f"note\tmemmap_failed\t{type(e).__name__}: {e}")
            # Fallback: ask tifffile to provide an array (may load into RAM)
            arr = series.asarray()

        if arr.ndim == 2:
            h, w = int(arr.shape[0]), int(arr.shape[1])
            print(f"image_hw\t{h}\t{w}")
            sample, step = _sample_view(arr, config.max_samples_per_band)
            _print_stats_for_band(sample, step=step, band_label="band0", config=config)
            return

        # Use axes to locate band dimension when possible
        band_axis = _infer_band_axis(axes)
        band_dim = None
        if band_axis and axes:
            band_dim = axes.index(band_axis)

        # Identify H/W from axes if present; else guess last two dims are Y,X
        if axes and "Y" in axes and "X" in axes:
            y_dim = axes.index("Y")
            x_dim = axes.index("X")
            h = int(arr.shape[y_dim])
            w = int(arr.shape[x_dim])
        else:
            h = int(arr.shape[-2])
            w = int(arr.shape[-1])
        print(f"image_hw\t{h}\t{w}")

        if band_dim is None:
            # If no band axis found, treat as single band (maybe extra dims like Z/T)
            sample, step = _sample_view(arr, config.max_samples_per_band)
            _print_stats_for_band(sample, step=step, band_label="band0", config=config)
            return

        num_bands = int(arr.shape[band_dim])
        print(f"num_bands\t{num_bands}")

        for b in range(num_bands):
            slicer = [slice(None)] * arr.ndim
            slicer[band_dim] = b
            band = arr[tuple(slicer)]
            sample, step = _sample_view(np.asarray(band), config.max_samples_per_band)
            _print_stats_for_band(
                sample,
                step=step,
                band_label=_band_name(band_axis, b),
                config=config,
            )


def _print_stats_for_band(
    sample_flat: np.ndarray,
    *,
    step: Optional[int],
    band_label: str,
    config: TiffStatsConfig,
) -> None:
    # Ensure 1D
    sample_flat = np.asarray(sample_flat).reshape(-1)

    # For stats, ignore NaNs if float
    is_float = np.issubdtype(sample_flat.dtype, np.floating)
    if is_float:
        finite = sample_flat[np.isfinite(sample_flat)]
        data = finite if finite.size else sample_flat
    else:
        data = sample_flat

    min_v = float(np.min(data)) if data.size else float("nan")
    max_v = float(np.max(data)) if data.size else float("nan")
    mean_v = float(np.mean(data)) if data.size else float("nan")
    std_v = float(np.std(data)) if data.size else float("nan")

    sampled_note = "full" if step is None else f"sampled(step={step})"
    print(
        "\t".join(
            [
                f"{band_label}",
                sampled_note,
                f"min={_format_number(min_v)}",
                f"max={_format_number(max_v)}",
                f"mean={_format_number(mean_v)}",
                f"std={_format_number(std_v)}",
            ]
        )
    )

    # Unique preview for integer-like data (useful for masks)
    if np.issubdtype(data.dtype, np.integer) and data.size:
        try:
            uniq = np.unique(data)
            if uniq.size <= config.max_unique_to_list:
                preview = uniq[: config.max_unique_preview]
                preview_str = ",".join(str(int(x)) for x in preview.tolist())
                suffix = "" if uniq.size <= config.max_unique_preview else f"...(+{uniq.size - config.max_unique_preview})"
                print(f"{band_label}\tunique\t{uniq.size}\t[{preview_str}{suffix}]")
            else:
                print(f"{band_label}\tunique\t{uniq.size}\t(too many to list)")
        except Exception as e:
            print(f"{band_label}\tunique\tERROR\t{e}")


if __name__ == "__main__":
    # This module is intended to be imported and called.
    # Example (run in Python):
    print_tiff_stats('/cluster/scratch/pangyi/reto/data/train/train_mask/gt_mask_map1.tif')
    raise SystemExit(
        "Import and call `print_tiff_stats(path)` from this file. "
        "(No default CLI in this script.)"
    )
