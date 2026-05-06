#!/usr/bin/env python3

import argparse
import os
import re

from make_merged_patches import cut_test_map, write_csv


def main():
    parser = argparse.ArgumentParser(description="Cut final competition test maps into SegFormer patches only.")
    parser.add_argument("--test-root", type=str, default="data_competition_original/test_final")
    parser.add_argument("--output-root", type=str, default="data")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    args = parser.parse_args()

    segformer_root = os.path.join(args.output_root, f"patches_{args.patch_size}_split")
    test_img_dir = os.path.join(segformer_root, "images", "test")
    os.makedirs(test_img_dir, exist_ok=True)

    test_images = []
    for fname in sorted(os.listdir(args.test_root)):
        m = re.match(r"^rgb_(map\d+)\.png$", fname)
        if m:
            test_images.append((fname, m.group(1)))

    if not test_images:
        raise FileNotFoundError(f"No rgb_map*.png found in {args.test_root}")

    all_rows = []
    for rgb_name, map_id in test_images:
        rgb_path = os.path.join(args.test_root, rgb_name)
        rows = cut_test_map(
            rgb_path=rgb_path,
            map_id=map_id,
            out_root=segformer_root,
            patch_size=args.patch_size,
            stride=args.stride,
        )
        all_rows.extend(rows)

    test_csv = os.path.join(segformer_root, "patch_index_test.csv")
    write_csv(
        test_csv,
        all_rows,
        fieldnames=[
            "patch_name",
            "source",
            "map_id",
            "y",
            "x",
            "height",
            "width",
            "image_path",
        ],
    )

    print(f"Saved final test patches to: {test_img_dir}")
    print(f"Saved final test patch index to: {test_csv}")
    print(f"Total test patches: {len(all_rows)}")


if __name__ == "__main__":
    main()
