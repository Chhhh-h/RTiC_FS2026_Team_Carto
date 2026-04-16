#!/usr/bin/env python3
"""Inspect .npy mask files and summarize class labels.

Usage:
  python scripts/inspect_mask_npy.py --mask-dir /cluster/scratch/pangyi/reto/data/data_augmentation/train/masks

This script prints:
- number of mask files
- array shape / dtype summary
- unique label values and pixel counts
- per-file examples
- whether labels appear to be binary, multiclass, or one-hot encoded
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# 直接在代码中指定路径和参数，不使用命令行 args。
MASK_DIR = Path("/cluster/scratch/pangyi/reto/data/data_augmentation/train/masks")
MAX_SAMPLES = 5
LIMIT_FILES = 0  # 0 表示全部文件；>0 表示只检查前 N 个文件


def load_mask(path: Path) -> np.ndarray:
    arr = np.load(path)
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{path} did not load as a numpy array")
    return arr


def summarize_array(arr: np.ndarray) -> Dict[str, object]:
    vals, counts = np.unique(arr, return_counts=True)
    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "min": arr.min().item() if arr.size else None,
        "max": arr.max().item() if arr.size else None,
        "unique_values": vals.tolist(),
        "unique_counts": counts.tolist(),
    }


def infer_encoding(values: List[int], arr_shape: Tuple[int, ...]) -> str:
    unique = sorted(set(values))
    if len(arr_shape) == 3 and arr_shape[-1] > 1:
        return "可能是 one-hot / multi-channel mask"
    if unique == [0, 1]:
        return "二分类 mask（0/1）"
    if unique and unique[0] == 0 and all(int(v) == v for v in unique):
        return "单通道多类 mask（像素值是类别 ID）"
    return "未知编码，需要结合数据再确认"


def main() -> None:
    mask_dir = MASK_DIR
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    files = sorted(mask_dir.glob("*.npy"))
    if LIMIT_FILES and LIMIT_FILES > 0:
        files = files[:LIMIT_FILES]

    if not files:
        print(f"No .npy files found in {mask_dir}")
        return

    global_value_counter: Counter = Counter()
    shape_counter: Counter = Counter()
    dtype_counter: Counter = Counter()
    sample_reports: List[Tuple[str, Dict[str, object], str]] = []

    for idx, file_path in enumerate(files):
        arr = load_mask(file_path)
        summary = summarize_array(arr)

        shape_counter[summary["shape"]] += 1
        dtype_counter[summary["dtype"]] += 1
        global_value_counter.update(
            {int(v): int(c) for v, c in zip(summary["unique_values"], summary["unique_counts"])}
        )

        if idx < MAX_SAMPLES:
            encoding = infer_encoding(summary["unique_values"], summary["shape"])
            sample_reports.append((file_path.name, summary, encoding))

    print("=" * 80)
    print(f"Mask directory: {mask_dir}")
    print(f"Total files inspected: {len(files)}")
    print("\n[Shape distribution]")
    for shape, count in shape_counter.most_common():
        print(f"  {shape}: {count}")

    print("\n[Dtype distribution]")
    for dtype, count in dtype_counter.most_common():
        print(f"  {dtype}: {count}")

    print("\n[Global label statistics]")
    print(f"  unique class values: {sorted(global_value_counter.keys())}")
    print(f"  number of classes/labels: {len(global_value_counter)}")
    print("  pixel counts per label:")
    for label, count in sorted(global_value_counter.items(), key=lambda x: x[0]):
        print(f"    {label}: {count}")

    print("\n[Sample files]")
    for name, summary, encoding in sample_reports:
        print(f"  File: {name}")
        print(f"    shape: {summary['shape']}")
        print(f"    dtype: {summary['dtype']}")
        print(f"    min/max: {summary['min']} / {summary['max']}")
        print(f"    unique_values: {summary['unique_values']}")
        print(f"    encoding_guess: {encoding}")

    print("\n[Summary]")
    print("  如果 unique_values 是像 0,1,2,3... 这样的整数，通常表示单通道语义分割标签，")
    print("  每个像素的数值就是类别 ID。")
    print("  如果数组形状是 (H, W, C) 且 C > 1，可能是 one-hot/多通道标注。")
    print("=" * 80)


if __name__ == "__main__":
    main()
