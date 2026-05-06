#!/usr/bin/env python3

import os
from itertools import combinations

import numpy as np
import tifffile
import matplotlib.pyplot as plt


MASK_PATHS = [
    'data_competition_original/train/gt_mask_map1.tif',
    'data_competition_original/train/gt_mask_map2.tif',
]

CLASS_NAMES = [
    'river',
    'forest',
    'lake',
    'wetland',
    'stream',
    'building',
    'road',
]

OUTPUT_DIR = 'debug/class_overlap'


def ensure_hwc_7(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 3:
        raise ValueError(f'Expected 3D mask, got shape {mask.shape}')
    if mask.shape[-1] == 7:
        return mask
    if mask.shape[0] == 7:
        return np.transpose(mask, (1, 2, 0))
    raise ValueError(f'Expected 7 channels, got shape {mask.shape}')


def load_mask(mask_path: str) -> np.ndarray:
    return tifffile.imread(mask_path)


def summarize_mask(mask_path: str):
    mask = load_mask(mask_path)
    mask = (ensure_hwc_7(mask) > 0).astype(np.uint8)

    print(f'\n=== {mask_path} ===')
    print(f'shape: {mask.shape}, total labeled pixels (multi-label counted per class): {int(mask.sum())}')

    pair_results = []
    positive_pairs = []
    zero_pairs = []

    for i, j in combinations(range(len(CLASS_NAMES)), 2):
        overlap = int(np.logical_and(mask[:, :, i] == 1, mask[:, :, j] == 1).sum())
        union = int(np.logical_or(mask[:, :, i] == 1, mask[:, :, j] == 1).sum())
        pair_name = f'{CLASS_NAMES[i]} <-> {CLASS_NAMES[j]}'
        pair_results.append((pair_name, overlap, union))
        if overlap > 0:
            positive_pairs.append((pair_name, overlap))
        else:
            zero_pairs.append(pair_name)

    print('\nPer-class positive pixels:')
    for idx, class_name in enumerate(CLASS_NAMES):
        pixels = int(mask[:, :, idx].sum())
        print(f'  {class_name:>8}: {pixels}')

    print('\nPairs with overlap:')
    if positive_pairs:
        for pair_name, overlap in sorted(positive_pairs, key=lambda x: (-x[1], x[0])):
            print(f'  {pair_name}: {overlap}')
    else:
        print('  None')

    print('\nPairs with no overlap:')
    if zero_pairs:
        for pair_name in zero_pairs:
            print(f'  {pair_name}')
    else:
        print('  None')

    return mask


def summarize_combined(masks):
    combined = np.concatenate(masks, axis=0)
    print('\n=== Combined map1 + map2 ===')
    print(f'shape: {combined.shape}')

    print('\nPer-class positive pixels:')
    for idx, class_name in enumerate(CLASS_NAMES):
        pixels = int(combined[:, :, idx].sum())
        print(f'  {class_name:>8}: {pixels}')

    positive_pairs = []
    zero_pairs = []
    for i, j in combinations(range(len(CLASS_NAMES)), 2):
        overlap = int(np.logical_and(combined[:, :, i] == 1, combined[:, :, j] == 1).sum())
        pair_name = f'{CLASS_NAMES[i]} <-> {CLASS_NAMES[j]}'
        if overlap > 0:
            positive_pairs.append((pair_name, overlap))
        else:
            zero_pairs.append(pair_name)

    print('\nPairs with overlap:')
    if positive_pairs:
        for pair_name, overlap in sorted(positive_pairs, key=lambda x: (-x[1], x[0])):
            print(f'  {pair_name}: {overlap}')
    else:
        print('  None')

    return combined


def compute_overlap_ratio_matrix(mask: np.ndarray) -> np.ndarray:
    num_classes = mask.shape[-1]
    matrix = np.zeros((num_classes, num_classes), dtype=np.float32)

    for i in range(num_classes):
        mask_i = mask[:, :, i] == 1
        for j in range(num_classes):
            mask_j = mask[:, :, j] == 1
            overlap = np.logical_and(mask_i, mask_j).sum()
            union = np.logical_or(mask_i, mask_j).sum()
            ratio = float(overlap) / float(union) if union > 0 else 0.0
            matrix[i, j] = ratio

    return matrix


def save_overlap_heatmap(matrix: np.ndarray, out_path: str):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    plot_matrix = matrix.copy()
    np.fill_diagonal(plot_matrix, np.nan)
    off_diag_max = np.nanmax(plot_matrix)
    if not np.isfinite(off_diag_max) or off_diag_max <= 0:
        off_diag_max = 1.0

    cmap = plt.cm.get_cmap('YlOrRd').copy()
    cmap.set_bad(color='white')

    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    im = ax.imshow(plot_matrix, cmap=cmap, vmin=0.0, vmax=off_diag_max)

    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_title('Class Overlap Ratio Heatmap\n(overlap / union, off-diagonal rescaled)')

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if i == j:
                continue
            permille = int(round(value * 1000.0))
            text_color = 'white' if value > 0.55 * off_diag_max else 'black'
            ax.text(j, i, f'{permille}\u2030', ha='center', va='center', color=text_color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Overlap Ratio (off-diagonal rescaled)')

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved overlap heatmap: {out_path}')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    masks = []
    for mask_path in MASK_PATHS:
        if not os.path.exists(mask_path):
            raise FileNotFoundError(mask_path)
        masks.append(summarize_mask(mask_path))

    combined = summarize_combined(masks)
    overlap_matrix = compute_overlap_ratio_matrix(combined)
    save_overlap_heatmap(
        overlap_matrix,
        os.path.join(OUTPUT_DIR, 'class_overlap_heatmap.png'))


if __name__ == '__main__':
    main()
