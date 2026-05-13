#!/usr/bin/env python3
# Cut full historical-map images into overlapping inference patches.

import argparse
import csv
import shutil
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Create inference patches and patch index for histmap images.")
    parser.add_argument("--dataset-root", default="/cluster/scratch/caizhi/vectorization/dataset", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-root", default="data/histmap_instance", type=Path)
    parser.add_argument("--output-split", default=None)
    parser.add_argument("--patch-size", default=1024, type=int)
    parser.add_argument("--stride", default=768, type=int)
    parser.add_argument("--jpeg-quality", default=95, type=int)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def iter_windows(width, height, patch_size, stride):
    xs = list(range(0, max(width - patch_size, 0) + 1, stride)) or [0]
    ys = list(range(0, max(height - patch_size, 0) + 1, stride)) or [0]
    if xs[-1] != max(width - patch_size, 0):
        xs.append(max(width - patch_size, 0))
    if ys[-1] != max(height - patch_size, 0):
        ys.append(max(height - patch_size, 0))
    for y in ys:
        for x in xs:
            yield x, y, min(patch_size, width - x), min(patch_size, height - y)


def main():
    args = parse_args()
    out_split = args.output_split or args.split
    split_dir = args.dataset_root / args.split
    image_dir = args.output_root / "images" / out_split
    index_path = args.output_root / f"patch_index_{out_split}.csv"

    if args.clean and image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_path in sorted(split_dir.glob("*-INPUT.jpg")):
        sample_id = image_path.name.split("-INPUT.jpg")[0]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        height, width = image.shape[:2]
        for x, y, w, h in iter_windows(width, height, args.patch_size, args.stride):
            file_name = f"{sample_id}__x{x:05d}_y{y:05d}.jpg"
            patch = image[y:y + h, x:x + w]
            cv2.imwrite(str(image_dir / file_name), patch, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
            rows.append({
                "sample_id": sample_id,
                "file_name": file_name,
                "patch_path": str(image_dir / file_name),
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "image_width": width,
                "image_height": height,
            })

    fieldnames = ["sample_id", "file_name", "patch_path", "x", "y", "width", "height", "image_width", "image_height"]
    with index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved {len(rows)} patches to {image_dir}")
    print(f"saved patch index to {index_path}")


if __name__ == "__main__":
    main()
