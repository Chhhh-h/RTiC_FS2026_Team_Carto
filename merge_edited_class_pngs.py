#!/usr/bin/env python3

import os

import numpy as np
from PIL import Image

from mask_to_submission import masks_to_submission_rle


INPUT_DIR = 'predictions_map3_fused/class_pngs'
OUTPUT_NPY = 'predictions_map3_fused/stitched_binary_map3_edited.npy'
OUTPUT_CSV = 'submission_fused_edited.csv'

# Channel order must match the submission class order.
CLASS_FILES = [
    'class_1.png',
    'class_2.png',
    'class_3.png',
    'class_4.png',
    'class_5.png',
    'class_6.png',
    'class_7.png',
]


def load_binary_png(path: str) -> np.ndarray:
    arr = np.array(Image.open(path).convert('L'), dtype=np.uint8)
    return (arr > 127).astype(np.uint8)


def main():
    masks = []
    shape = None

    for name in CLASS_FILES:
        path = os.path.join(INPUT_DIR, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f'Missing edited PNG: {path}')

        mask = load_binary_png(path)
        if shape is None:
            shape = mask.shape
        elif mask.shape != shape:
            raise ValueError(f'Shape mismatch: expected {shape}, got {mask.shape} for {path}')

        masks.append(mask)
        print(f'Loaded {name}: shape={mask.shape}, pixels={int(mask.sum())}')

    stitched = np.stack(masks, axis=0).astype(np.uint8)

    overlap = stitched.sum(axis=0)
    overlap_pixels = int((overlap > 1).sum())
    empty_pixels = int((overlap == 0).sum())

    print(f'Combined mask shape: {stitched.shape}')
    print(f'Overlap pixels (>1 class): {overlap_pixels}')
    print(f'Empty pixels (no class): {empty_pixels}')

    if overlap_pixels > 0:
        print('Warning: overlapping class labels found. Keeping them as-is in the output mask.')

    os.makedirs(os.path.dirname(OUTPUT_NPY) or '.', exist_ok=True)
    np.save(OUTPUT_NPY, stitched)
    print(f'Saved merged npy: {OUTPUT_NPY}')

    masks_to_submission_rle(OUTPUT_CSV, stitched)
    print(f'Saved submission csv: {OUTPUT_CSV}')


if __name__ == '__main__':
    main()
