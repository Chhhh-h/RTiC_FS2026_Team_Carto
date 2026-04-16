#!/usr/bin/env python3

import cv2

import numpy as np
from pycocotools import mask as maskUtils


def mask_to_rle(binary_mask):
    """Encode a binary mask as a COCO RLE string using pycocotools."""
    fortran_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = maskUtils.encode(fortran_mask)
    return rle['counts'].decode('utf-8')


def mask_to_submission_strings_rle(mask: np.ndarray):
    """Yields per-class RLE strings from a C×H×W binary mask array.

    Args:
        mask: 3-D numpy array of shape (C, H, W) where C is the number of
              classes and each channel is a binary mask for that class.

    Yields:
        Strings of the form '{class_id},{rle_string}'.
    """
    for cls_id, binary in enumerate(mask): # 0-7
        rle = mask_to_rle(binary.astype(np.uint8))
        yield("{},{}".format(cls_id, rle))


def masks_to_submission_rle(submission_filename, mask: np.ndarray):
    """Converts a C×H×W binary mask array into a submission CSV using per-class RLE encoding.

    Args:
        submission_filename: Path to the output CSV file.
        mask:                3-D numpy array of shape (C, H, W)
                             where each channel is a binary mask for that class.
    """
    with open(submission_filename, 'w') as f:
        f.write('ID,rle\n')
        f.writelines('{}\n'.format(s) for s in mask_to_submission_strings_rle(mask))

def load_binary_masks_from_png(mask_dir: str, num_classes: int):
    """Load class_0.png ... class_{C-1}.png into a C×H×W array."""
    masks = []
    for cls_id in range(num_classes):
        path = f'{mask_dir}/class_{cls_id}.png'
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f'Cannot read image: {path}')

        
        mask = (mask > 0).astype(np.uint8)
        # check whether 0/1 binary
        uniq = np.unique(mask)
        if not np.all(np.isin(uniq, [0, 1])):
            raise ValueError(
                f'Channel class_{cls_id} is not 0/1 binary. '
                f'Found values: {uniq[:20]}'
            )
        masks.append(mask)

    chw_mask = np.stack(masks, axis=0)
    return chw_mask

if __name__ == '__main__':
    mask_dir = '/cluster/scratch/pangyi/reto/submission_result/expt5'
    num_classes = 8
    chw_mask = load_binary_masks_from_png(mask_dir, num_classes)
    print('chw_mask shape:', chw_mask.shape)
    print('chw_mask dtype:', chw_mask.dtype)

    # H, W = 9600, 14000
    # # Your prediction of binary mask for each category
    # pred_mask_river = np.zeros((H,W))
    # pred_mask_building = np.zeros((H,W))
    # # Stack all the predictions into a CxHxW np.array where C is the category number
    # chw_mask = np.stack([...,pred_mask_river, pred_mask_building,...]) # 7xHxW
    # Write the array to submission.csv file
    masks_to_submission_rle('/cluster/scratch/pangyi/reto/submission_result/expt5/submission.csv', chw_mask)
