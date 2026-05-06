#!/usr/bin/env python3

import argparse
import numpy as np
from pycocotools import mask as maskUtils


def mask_to_rle(binary_mask: np.ndarray) -> str:
    """Encode a binary mask as a COCO RLE string using pycocotools."""
    fortran_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = maskUtils.encode(fortran_mask)
    return rle["counts"].decode("utf-8")


def mask_to_submission_strings_rle(mask: np.ndarray):
    """Yield per-class RLE strings from a CxHxW binary mask array.

    Args:
        mask: numpy array of shape (C, H, W), values should be 0/1.

    Yields:
        Strings like:
            '1,<rle>'
            '2,<rle>'
            ...
    """
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3D [C,H,W], but got shape {mask.shape}")

    num_classes = mask.shape[0]

    for cls_id in range(num_classes):
        binary = mask[cls_id]
        rle = mask_to_rle(binary)
        yield f"{cls_id + 1},{rle}"


def masks_to_submission_rle(submission_filename: str, mask: np.ndarray):
    """Convert a CxHxW binary mask array into a submission CSV."""
    with open(submission_filename, "w", encoding="utf-8") as f:
        f.write("ID,rle\n")
        for s in mask_to_submission_strings_rle(mask):
            f.write(f"{s}\n")


def main():
    parser = argparse.ArgumentParser(description="Convert stitched prediction to submission CSV")
    parser.add_argument(
        "--input-mask",
        type=str,
        required=True,
        help="Path to stitched .npy prediction, shape [C,H,W]",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="submission.csv",
        help="Output submission CSV path",
    )
    parser.add_argument(
        "--from-prob",
        action="store_true",
        help="Use this if input is probability map instead of binary map",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used when --from-prob is set",
    )
    args = parser.parse_args()

    mask = np.load(args.input_mask)

    if mask.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got shape {mask.shape}")

    # 如果读进来是概率图，就先阈值化
    if args.from_prob:
        mask = (mask > args.threshold).astype(np.uint8)
    else:
        mask = (mask > 0).astype(np.uint8)

    print(f"Loaded mask shape: {mask.shape}")
    print(f"Writing submission to: {args.output_csv}")

    masks_to_submission_rle(args.output_csv, mask)

    print("Done.")


if __name__ == "__main__":
    main()
