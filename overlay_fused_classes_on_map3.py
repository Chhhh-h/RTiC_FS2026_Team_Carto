#!/usr/bin/env python3

import os

import numpy as np
from PIL import Image


BASE_IMAGE_ROOT = 'data_competition_original/test_final'
CLASS_PNG_ROOT = 'predictions_final_fused'
OUTPUT_ROOT = 'predictions_final_fused'
MAP_IDS = ['map3', 'map4']

CLASS_FILES = [
    ('class_1.png', 'river', (0, 0, 255)),
    ('class_2.png', 'forest', (34, 139, 34)),
    ('class_3.png', 'lake', (0, 255, 255)),
    ('class_4.png', 'wetland', (255, 0, 255)),
    ('class_5.png', 'stream', (135, 206, 250)),
    ('class_6.png', 'building', (255, 165, 0)),
    ('class_7.png', 'road', (255, 255, 0)),
]

ALPHA = 0.4


def load_binary_mask(path: str) -> np.ndarray:
    mask = np.array(Image.open(path).convert('L'), dtype=np.uint8)
    return mask > 0


def make_overlay_for_map(map_id: str):
    base_image_path = os.path.join(BASE_IMAGE_ROOT, f'rgb_{map_id}.png')
    class_png_dir = os.path.join(CLASS_PNG_ROOT, map_id)
    output_path = os.path.join(OUTPUT_ROOT, f'{map_id}_overlay.png')

    base = np.array(Image.open(base_image_path).convert('RGB'), dtype=np.uint8)
    overlay = base.astype(np.float32).copy()

    print(f'\n=== {map_id} ===')
    print(f'base image: {base_image_path}')
    print(f'base shape: {base.shape}')

    for filename, class_name, color in CLASS_FILES:
        path = os.path.join(class_png_dir, filename)
        if not os.path.exists(path):
            print(f'skip missing: {path}')
            continue

        mask = load_binary_mask(path)
        if mask.shape != base.shape[:2]:
            raise ValueError(
                f'Size mismatch for {path}: mask={mask.shape}, base={base.shape[:2]}'
            )

        color_arr = np.array(color, dtype=np.float32)
        overlay[mask] = overlay[mask] * (1.0 - ALPHA) + color_arr * ALPHA
        print(f'{class_name}: pixels={int(mask.sum())}, file={path}')

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    Image.fromarray(overlay, mode='RGB').save(output_path)
    print(f'saved overlay to: {output_path}')


def main():
    for map_id in MAP_IDS:
        make_overlay_for_map(map_id)


if __name__ == '__main__':
    main()
