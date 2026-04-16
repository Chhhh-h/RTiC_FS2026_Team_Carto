# give each patch a weight according to class rarity so rare-class patches are sampled more frequently during training.
import argparse
import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

IGNORE_INDEX = 255
NUM_CLASSES = 8

class_freq = np.array([
    0.614457,  # background
    0.016369,  # river
    0.305113,  # forest
    0.004246,  # lake
    0.001058,  # wetland
    0.004647,  # stream
    0.006650,  # building
    0.047460,  # road
], dtype=np.float32)

class_rarity = 1.0 / np.sqrt(class_freq + 1e-6)
class_rarity = class_rarity / class_rarity.mean() # normalization

def compute_patch_weight(mask_path):
    mask = np.array(Image.open(mask_path))
    valid = mask != IGNORE_INDEX
    if valid.sum() == 0:
        return 1.0

    labels = mask[valid]

    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    ratios = counts / counts.sum() # get the class ratio of this patch

    # get rid off the background class
    ratios[0] = 0.0

    weight = 1.0 + np.sum(ratios * (class_rarity ** 1.5)) # weight = 1.0 + contibution of all classes in this patch (sum of  class ratio * class rarity)
    return float(weight)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate per-sample weights for WeightedRandomSampler.')
    parser.add_argument(
        '--img-dir',
        default='/cluster/scratch/pangyi/reto/data/patches_640_split/images/training',
        help='Training image directory. Used to define sample order.')
    parser.add_argument(
        '--mask-dir',
        default='/cluster/scratch/pangyi/reto/data/patches_640_split/annotations/training_background',
        help='Training annotation directory.')
    parser.add_argument(
        '--img-suffix', default='.png', help='Image file suffix.')
    parser.add_argument(
        '--mask-suffix', default='.tif', help='Mask file suffix.')
    parser.add_argument(
        '--output',
        default='/cluster/scratch/pangyi/reto/data/patches_640_split/sample_weights.npy',
        help='Output .npy path for per-sample weights.')
    parser.add_argument(
        '--metadata-output',
        default='/cluster/scratch/pangyi/reto/data/patches_640_split/sample_weights_meta.json',
        help='Output metadata json path (filenames and stats).')
    return parser.parse_args()


def main():
    args = parse_args()

    image_files = sorted([
        f for f in os.listdir(args.img_dir)
        if f.endswith(args.img_suffix)
    ])

    if not image_files:
        raise RuntimeError(f'No images found in {args.img_dir}')

    patch_weights = []
    missing_masks = []
    for image_name in tqdm(image_files, desc='Computing sample weights'):
        mask_name = image_name.replace(args.img_suffix, args.mask_suffix)
        mask_path = os.path.join(args.mask_dir, mask_name)
        if not os.path.exists(mask_path):
            missing_masks.append(mask_name)
            continue
        patch_weights.append(compute_patch_weight(mask_path))

    if missing_masks:
        raise FileNotFoundError(
            f'{len(missing_masks)} masks are missing, first 5: '
            f'{missing_masks[:5]}')

    patch_weights = np.array(patch_weights, dtype=np.float32)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, patch_weights)

    stats = {
        'num_samples': int(patch_weights.size),
        'min': float(patch_weights.min()),
        'max': float(patch_weights.max()),
        'mean': float(patch_weights.mean()),
        'weights_file': args.output,
        'img_dir': args.img_dir,
        'mask_dir': args.mask_dir,
        'img_suffix': args.img_suffix,
        'mask_suffix': args.mask_suffix,
        'filenames': image_files,
    }
    os.makedirs(os.path.dirname(args.metadata_output), exist_ok=True)
    with open(args.metadata_output, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(
        f"Saved {patch_weights.size} weights to {args.output}. "
        f"min={stats['min']:.4f}, max={stats['max']:.4f}, mean={stats['mean']:.4f}")
    print(f'Saved metadata to {args.metadata_output}')


if __name__ == '__main__':
    main()
