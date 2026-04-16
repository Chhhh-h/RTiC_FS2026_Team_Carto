#!/usr/bin/env python3
"""Tile paired RGB images and segmentation masks into fixed-size patches.

RGB patches and corresponding mask patches have identical names.
Mask background value is 255.
"""

from __future__ import annotations

import csv
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Pair:
    rgb_path: Path
    mask_path: Path
    pair_id: str


def _disable_pillow_decompression_bomb_warning() -> None:
    # These maps are very large (e.g. 9600x14000). We intentionally load them.
    Image.MAX_IMAGE_PIXELS = None


def _load_rgb(path: Path) -> np.ndarray:
    _disable_pillow_decompression_bomb_warning()
    arr = np.array(Image.open(path))
    
    # Handle RGBA (4 channels) by dropping alpha channel
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"RGB image must be HxWx3, got shape={arr.shape} for {path}")
    return arr


def _load_mask(path: Path) -> np.ndarray:
    # Prefer tifffile for TIFF to avoid mode surprises.
    try:
        import tifffile as tiff  # type: ignore

        arr = tiff.imread(str(path))
    except Exception:
        _disable_pillow_decompression_bomb_warning()
        arr = np.array(Image.open(path))

    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    if arr.ndim != 2:
        raise ValueError(f"Mask must be HxW (single-channel), got shape={arr.shape} for {path}")
    return arr


def _save_mask_tif(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import tifffile as tiff  # type: ignore

        tiff.imwrite(str(path), mask)
    except Exception:
        # Fallback: Pillow can write TIFF, but may change metadata.
        Image.fromarray(mask).save(path)


def _save_rgb_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def _infer_pair_id_from_rgb(rgb_path: Path) -> str:
    # rgb_map1.png -> map1
    stem = rgb_path.stem
    if stem.startswith("rgb_"):
        return stem[len("rgb_") :]
    return stem


def _find_mask_for_pair_id(mask_dir: Path, pair_id: str) -> Optional[Path]:
    # Expected: gt_mask_<id>.tif
    candidate = mask_dir / f"gt_mask_{pair_id}.tif"
    if candidate.exists():
        return candidate

    # Fallback: any tif that contains the id (useful if naming differs slightly)
    matches = sorted(mask_dir.glob(f"*{pair_id}*.tif"))
    if len(matches) == 1:
        return matches[0]
    return None


def discover_pairs(image_dir: Path, mask_dir: Path) -> list[Pair]:
    rgb_paths = sorted(
        [
            *image_dir.glob("rgb_*.png"),
            *image_dir.glob("rgb_*.jpg"),
            *image_dir.glob("rgb_*.jpeg"),
        ]
    )

    pairs: list[Pair] = []
    for rgb_path in rgb_paths:
        pair_id = _infer_pair_id_from_rgb(rgb_path)
        mask_path = _find_mask_for_pair_id(mask_dir, pair_id)
        if mask_path is None:
            raise FileNotFoundError(
                f"No unique mask found for RGB={rgb_path} (pair_id={pair_id}) in {mask_dir}"
            )
        pairs.append(Pair(rgb_path=rgb_path, mask_path=mask_path, pair_id=pair_id))

    if not pairs:
        raise FileNotFoundError(f"No RGB images found in {image_dir} matching rgb_*.png/jpg")
    return pairs


def _compute_needed_pad(length: int, tile: int, stride: int) -> int:
    if length <= tile:
        needed = tile
    else:
        n_steps = int(math.ceil((length - tile) / stride))
        needed = n_steps * stride + tile
    return needed - length


# Pad the original image to ensure integer number of patches
def _pad_pair(
    rgb: np.ndarray, mask: np.ndarray, tile: int, stride: int, pad_value_rgb: int = 0, pad_value_mask: int = 255 # Padding value for mask, corresponding to background class 255
) -> Tuple[np.ndarray, np.ndarray]:
    if rgb.shape[0] != mask.shape[0] or rgb.shape[1] != mask.shape[1]:
        raise ValueError(f"RGB/mask size mismatch: rgb={rgb.shape}, mask={mask.shape}")

    pad_h = _compute_needed_pad(rgb.shape[0], tile, stride)
    pad_w = _compute_needed_pad(rgb.shape[1], tile, stride)

    if pad_h == 0 and pad_w == 0:
        return rgb, mask

    rgb_pad = np.pad(
        rgb,
        pad_width=((0, pad_h), (0, pad_w), (0, 0)),
        mode="constant",
        constant_values=pad_value_rgb,
    )
    mask_pad = np.pad(
        mask,
        pad_width=((0, pad_h), (0, pad_w)),
        mode="constant",
        constant_values=pad_value_mask,
    )
    return rgb_pad, mask_pad

# Generate tile start coordinates, ensuring tiles are entirely within the image
def _iter_windows(h: int, w: int, tile: int, stride: int) -> Iterable[Tuple[int, int, int, int]]:
    # yields (row_idx, col_idx, y, x)
    if h < tile or w < tile:
        raise ValueError(f"Padded size must be >= tile. Got h={h}, w={w}, tile={tile}")

    n_rows = ((h - tile) // stride) + 1
    n_cols = ((w - tile) // stride) + 1
    for r in range(n_rows):
        y = r * stride
        for c in range(n_cols):
            x = c * stride
            yield r, c, y, x


# Main function to tile images, cut patches based on tile coordinates and save them
def tile_one_pair(
    pair: Pair,
    out_dir: Path,
    tile: int,
    stride: int,
    pad: bool,
    manifest_rows: list[dict],
    patch_counter: dict,
) -> dict:
    rgb = _load_rgb(pair.rgb_path)
    mask = _load_mask(pair.mask_path)

    if pad:
        rgb, mask = _pad_pair(rgb, mask, tile=tile, stride=stride)
    else:
        # Only keep full tiles that fit entirely.
        h, w = rgb.shape[0], rgb.shape[1]
        h = (h - tile) // stride * stride + tile if h >= tile else 0
        w = (w - tile) // stride * stride + tile if w >= tile else 0
        rgb = rgb[:h, :w]
        mask = mask[:h, :w]

    images_dir = out_dir / "images"
    masks_dir = out_dir / "annotations"

    patch_start = patch_counter["count"]
    patch_count_for_this_pair = 0

    for r, c, y, x in _iter_windows(rgb.shape[0], rgb.shape[1], tile=tile, stride=stride):
        rgb_patch = rgb[y : y + tile, x : x + tile, :]
        mask_patch = mask[y : y + tile, x : x + tile]

        patch_index = patch_counter["count"]
        patch_stem = f"patch_{patch_index:03d}"
        patch_counter["count"] += 1
        patch_count_for_this_pair += 1
        
        rgb_out = images_dir / f"{patch_stem}.png"
        mask_out = masks_dir / f"{patch_stem}.tif"

        _save_rgb_png(rgb_out, rgb_patch)
        _save_mask_tif(mask_out, mask_patch)

        manifest_rows.append(
            {
                "pair_id": pair.pair_id,
                "image_patch": str(rgb_out.as_posix()),
                "mask_patch": str(mask_out.as_posix()),
                "source_image": str(pair.rgb_path.as_posix()),
                "source_mask": str(pair.mask_path.as_posix()),
                "row": r,
                "col": c,
                "y": y,
                "x": x,
                "tile_size": tile,
                "stride": stride,
                "padded": int(pad),
            }
        )

    patch_end = patch_counter["count"] - 1
    return {
        "pair_id": pair.pair_id,
        "patch_start": patch_start,
        "patch_end": patch_end,
        "patch_count": patch_count_for_this_pair,
    }


def write_manifest_csv(out_dir: Path, rows: list[dict]) -> Path:
    out_path = out_dir / "manifest.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "pair_id",
        "image_patch",
        "mask_patch",
        "source_image",
        "source_mask",
        "row",
        "col",
        "y",
        "x",
        "tile_size",
        "stride",
        "padded",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return out_path


def split_train_val_by_pair(
    patches_dir: Path,
    out_dir: Path,
    manifest_path: Path,
    train_ratio: float = 0.8,
    seed: Optional[int] = 42,
) -> None:
    """Split patches into train/val sets, stratified by pair_id.
    
    For each pair_id, randomly select train_ratio for training and 
    (1-train_ratio) for validation. Copy files to:
      out_dir/images/training/
      out_dir/annotations/training/
      out_dir/images/validation/
      out_dir/annotations/validation/
    """
    if seed is not None:
        random.seed(seed)
    
    # Read manifest
    rows = []
    with manifest_path.open("r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        print(f"No patches found in {manifest_path}")
        return
    
    # Group by pair_id
    pair_groups = {}
    for row in rows:
        pair_id = row["pair_id"]
        if pair_id not in pair_groups:
            pair_groups[pair_id] = []
        pair_groups[pair_id].append(row)
    
    # Split each pair into train/val
    train_rows = []
    val_rows = []
    
    split_info = []
    for pair_id in sorted(pair_groups.keys()):
        group = pair_groups[pair_id]
        n_total = len(group)
        n_train = int(n_total * train_ratio)
        
        # Randomly shuffle and split
        shuffled = group.copy()
        random.shuffle(shuffled)
        
        train_group = shuffled[:n_train]
        val_group = shuffled[n_train:]
        
        train_rows.extend(train_group)
        val_rows.extend(val_group)
        
        split_info.append({
            "pair_id": pair_id,
            "total": n_total,
            "train": len(train_group),
            "val": len(val_group),
        })
    
    # Create output directories
    img_train_dir = out_dir / "images" / "training"
    ann_train_dir = out_dir / "annotations" / "training"
    img_val_dir = out_dir / "images" / "validation"
    ann_val_dir = out_dir / "annotations" / "validation"
    
    img_train_dir.mkdir(parents=True, exist_ok=True)
    ann_train_dir.mkdir(parents=True, exist_ok=True)
    img_val_dir.mkdir(parents=True, exist_ok=True)
    ann_val_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy training files
    for row in train_rows:
        src_img = patches_dir / row["image_patch"]
        src_ann = patches_dir / row["mask_patch"]
        dst_img = img_train_dir / src_img.name
        dst_ann = ann_train_dir / src_ann.name
        
        if src_img.exists():
            shutil.copy2(src_img, dst_img)
        if src_ann.exists():
            shutil.copy2(src_ann, dst_ann)
    
    # Copy validation files
    for row in val_rows:
        src_img = patches_dir / row["image_patch"]
        src_ann = patches_dir / row["mask_patch"]
        dst_img = img_val_dir / src_img.name
        dst_ann = ann_val_dir / src_ann.name
        
        if src_img.exists():
            shutil.copy2(src_img, dst_img)
        if src_ann.exists():
            shutil.copy2(src_ann, dst_ann)
    
    # Write new manifests
    train_manifest = out_dir / "manifest_train.csv"
    val_manifest = out_dir / "manifest_val.csv"
    
    with train_manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(train_rows)
    
    with val_manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(val_rows)
    
    # Print split info
    print("\n=== Train/Val Split (by pair_id) ===")
    for info in split_info:
        print(f"pair_id: {info['pair_id']:8s} | total: {info['total']:4d} | train: {info['train']:4d} (80%) | val: {info['val']:4d} (20%)")
    print(f"\nTotal train: {len(train_rows)} | Total val: {len(val_rows)}")
    print(f"Train manifest: {train_manifest}")
    print(f"Val manifest: {val_manifest}")



def main() -> int:
    # Hardcoded parameters
    image_dir = Path("/cluster/scratch/pangyi/reto/data/origin/train/train")
    mask_dir = Path("/cluster/scratch/pangyi/reto/data/origin/train/train_mask")
    patches_dir = Path("/cluster/scratch/pangyi/reto/data/patches_512")
    split_out_dir = Path("/cluster/scratch/pangyi/reto/data/patches_512_split")
    tile = 512
    stride = 512
    pad = True  # Use padding

    pairs = discover_pairs(image_dir, mask_dir)

    manifest_rows: list[dict] = []
    patch_counter = {"count": 0}
    pair_stats = []

    for pair in pairs:
        stats = tile_one_pair(pair, out_dir=patches_dir, tile=tile, stride=stride, pad=pad, manifest_rows=manifest_rows, patch_counter=patch_counter)
        pair_stats.append(stats)

    manifest_path = write_manifest_csv(patches_dir, manifest_rows)
    print(f"Wrote {len(manifest_rows)} patches. Manifest: {manifest_path}")
    print("\n=== Patch Distribution ===")
    for stats in pair_stats:
        print(f"pair_id: {stats['pair_id']:8s} | patches: {stats['patch_count']:4d} | indices: {stats['patch_start']:3d} - {stats['patch_end']:3d}")
    
    # Split into train/val
    print("\n" + "="*50)
    manifest_path = Path("/cluster/scratch/pangyi/reto/data/patches_512/manifest.csv")

    split_train_val_by_pair(patches_dir, split_out_dir, manifest_path, train_ratio=0.8)
    
    return 0


if __name__ == "__main__":
    main()

