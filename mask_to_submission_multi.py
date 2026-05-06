#!/usr/bin/env python3

import argparse
import csv
import os

import numpy as np
from pycocotools import mask as maskUtils


def mask_to_rle(binary_mask: np.ndarray) -> str:
    fortran_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = maskUtils.encode(fortran_mask)
    return rle["counts"].decode("utf-8")


def read_template_ids(template_csv: str):
    with open(template_csv, newline="", encoding="utf-8") as f:
        return [row["ID"] for row in csv.DictReader(f)]


def parse_template_id(item_id: str):
    map_id, cls_id = item_id.rsplit("_", 1)
    return map_id, int(cls_id)


def load_mask_for_map(input_dir: str, map_id: str, filename_template: str):
    path = os.path.join(input_dir, filename_template.format(map_id=map_id))
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    mask = np.load(path)
    if mask.ndim != 3:
        raise ValueError(f"Expected [C,H,W] mask at {path}, got shape {mask.shape}")
    return (mask > 0).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Convert multi-map stitched masks to submission CSV")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing per-map stitched .npy files",
    )
    parser.add_argument(
        "--template-csv",
        type=str,
        default="data_competition_original/test_final/submission_tempate.csv",
        help="Submission template with IDs like map3_1,map4_7",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Output submission CSV",
    )
    parser.add_argument(
        "--filename-template",
        type=str,
        default="{map_id}_stitched_binary.npy",
        help="Per-map stitched mask filename template",
    )
    args = parser.parse_args()

    template_ids = read_template_ids(args.template_csv)
    mask_cache = {}

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8") as f:
        f.write("ID,rle\n")
        for item_id in template_ids:
            map_id, cls_id = parse_template_id(item_id)
            if map_id not in mask_cache:
                mask_cache[map_id] = load_mask_for_map(
                    args.input_dir,
                    map_id,
                    args.filename_template)

            mask = mask_cache[map_id]
            if cls_id < 1 or cls_id > mask.shape[0]:
                raise ValueError(
                    f"Class id {cls_id} out of range for {map_id} mask shape {mask.shape}"
                )

            rle = mask_to_rle(mask[cls_id - 1])
            f.write(f"{item_id},{rle}\n")

    print(f"Saved submission to: {args.output_csv}")


if __name__ == "__main__":
    main()
