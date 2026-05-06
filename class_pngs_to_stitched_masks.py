#!/usr/bin/env python3

import argparse
import os

import numpy as np
from PIL import Image


DEFAULT_CLASS_FILES = [
    'class_1.png',
    'class_2.png',
    'class_3.png',
    'class_4.png',
    'class_5.png',
    'class_6.png',
    'class_7.png',
]


def load_binary_png(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    arr = np.array(Image.open(path).convert("L"))
    return (arr > 0).astype(np.uint8)


def build_mask_for_map(map_dir: str, class_files) -> np.ndarray:
    channels = [load_binary_png(os.path.join(map_dir, fname)) for fname in class_files]
    shapes = {channel.shape for channel in channels}
    if len(shapes) != 1:
        raise ValueError(f"Class PNG sizes are not all equal in {map_dir}: {sorted(shapes)}")
    return np.stack(channels, axis=0).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-class PNG masks into per-map stitched [7,H,W] .npy files."
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="predictions_final_fused",
        help="Root containing map subdirectories, e.g. map3/ and map4/",
    )
    parser.add_argument(
        "--map-ids",
        type=str,
        default="map3,map4",
        help="Comma-separated map ids to process",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Where to save .npy masks. Defaults to --input-root.",
    )
    parser.add_argument(
        "--output-template",
        type=str,
        default="{map_id}_stitched_binary.npy",
        help="Output filename template",
    )
    parser.add_argument(
        "--class-files",
        type=str,
        default=",".join(DEFAULT_CLASS_FILES),
        help="Comma-separated class PNG filenames in channel order",
    )
    args = parser.parse_args()

    output_root = args.output_root or args.input_root
    os.makedirs(output_root, exist_ok=True)

    map_ids = [x.strip() for x in args.map_ids.split(",") if x.strip()]
    class_files = [x.strip() for x in args.class_files.split(",") if x.strip()]

    for map_id in map_ids:
        map_dir = os.path.join(args.input_root, map_id)
        mask = build_mask_for_map(map_dir, class_files)
        out_path = os.path.join(output_root, args.output_template.format(map_id=map_id))
        np.save(out_path, mask)
        print(f"Saved {map_id}: {out_path}, shape={mask.shape}")


if __name__ == "__main__":
    main()
