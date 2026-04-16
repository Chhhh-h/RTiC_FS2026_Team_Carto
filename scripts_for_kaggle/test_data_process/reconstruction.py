import pickle
import json
from pathlib import Path
import numpy as np
import cv2
from pycocotools import mask as maskUtils


def load_predictions(pkl_path):
    with open(pkl_path, "rb") as f:
        predictions = pickle.load(f)
    print(f"Loaded {len(predictions)} predictions from {pkl_path}")
    return predictions


def load_metadata(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata


def compute_start_positions(total_size: int, patch_size: int, stride: int):
    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride}")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be > 0, got {patch_size}")
    if stride > patch_size:
        raise ValueError(f"stride ({stride}) must be <= patch_size ({patch_size})")

    max_start = total_size - patch_size
    starts = list(range(0, max_start + 1, stride))
    if starts[-1] != max_start:
        starts.append(max_start)
    return starts


def create_center_weight(patch_size: int):
    y = np.arange(patch_size, dtype=np.float32)
    x = np.arange(patch_size, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    cy = (patch_size - 1) / 2.0
    cx = (patch_size - 1) / 2.0
    weight = 1.0 / (1.0 + np.abs(yy - cy) + np.abs(xx - cx))
    return weight

def to_2d_pred(pred):
    """
    Convert a single patch prediction to a (H, W) class map.
    Supported:
    - (H, W)
    - (1, H, W)
    """
    pred = np.array(pred)

    if pred.ndim == 2:
        return pred
    elif pred.ndim == 3 and pred.shape[0] == 1:
        return pred[0]
    else:
        raise ValueError(f"Unsupported prediction shape: {pred.shape}")


def stitch_patches_from_pkl(
    predictions,
    orig_h,
    orig_w,
    patch_size=640,
    stride=640,
    metadata=None,
):
    """
    Stitch patch predictions from pkl back to the original image size.
    Assumptions:
    - Patch order = top to bottom, left to right within each row
    - When cutting patches, edges smaller than patch_size were zero-padded
    """

    if metadata is not None:
        pad_h = metadata["padded_height"]
        pad_w = metadata["padded_width"]
        if metadata.get("patch_size") != patch_size:
            raise ValueError(
                f"patch_size mismatch: metadata={metadata.get('patch_size')}, input={patch_size}"
            )
        if metadata.get("stride", stride) != stride:
            raise ValueError(
                f"stride mismatch: metadata={metadata.get('stride')}, input={stride}"
            )
        patch_positions = [(p["y_start"], p["x_start"]) for p in metadata["patches"]]
        num_rows = metadata.get("num_rows")
        num_cols = metadata.get("num_cols")
    else:
        pad_h = ((orig_h + patch_size - 1) // patch_size) * patch_size
        pad_w = ((orig_w + patch_size - 1) // patch_size) * patch_size
        y_starts = compute_start_positions(pad_h, patch_size, stride)
        x_starts = compute_start_positions(pad_w, patch_size, stride)
        patch_positions = [(y, x) for y in y_starts for x in x_starts]
        num_rows = len(y_starts)
        num_cols = len(x_starts)

    expected_num_patches = len(patch_positions)

    print(f"Original size: ({orig_h}, {orig_w})")
    print(f"Padded size:   ({pad_h}, {pad_w})")
    print(f"Grid:          {num_rows} x {num_cols}")
    print(f"Expected patches: {expected_num_patches}")
    print(f"Actual patches:   {len(predictions)}")

    if len(predictions) != expected_num_patches:
        raise ValueError(
            f"Patch count mismatch: expected {expected_num_patches}, got {len(predictions)}"
        )

    stitched = np.zeros((pad_h, pad_w), dtype=np.uint8)
    best_weight = np.zeros((pad_h, pad_w), dtype=np.float32)
    center_weight = create_center_weight(patch_size)

    pred_idx = 0
    for y0, x0 in patch_positions:
            y1 = y0 + patch_size
            x1 = x0 + patch_size

            pred_2d = to_2d_pred(predictions[pred_idx])

            if pred_2d.shape != (patch_size, patch_size):
                raise ValueError(
                    f"Prediction at index {pred_idx} has shape {pred_2d.shape}, "
                    f"expected ({patch_size}, {patch_size})"
                )

            current_best = best_weight[y0:y1, x0:x1]
            update_mask = center_weight > current_best
            target = stitched[y0:y1, x0:x1]
            target[update_mask] = pred_2d[update_mask]
            current_best[update_mask] = center_weight[update_mask]
            pred_idx += 1

    # Crop the padded area to recover original size
    final_map = stitched[:orig_h, :orig_w]
    return final_map


def extract_binary_masks(segmentation_map, num_classes):
    """
    Convert the full-image class labels to (C, H, W) binary masks.
    """
    h, w = segmentation_map.shape
    masks = np.zeros((num_classes, h, w), dtype=np.uint8)
    for cls_id in range(num_classes):
        masks[cls_id] = (segmentation_map == cls_id).astype(np.uint8)
    return masks


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
        Skips class 0.
    """
    for cls_id, binary in enumerate(mask): # 0-7
        if cls_id == 0:  # Skip class 0
            continue
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





def main():
    pkl_path = "/cluster/scratch/pangyi/reto/submission_result/expt15_b5_moreclass_4batch_160k/submission_results.pkl"
    output_dir = Path("/cluster/scratch/pangyi/reto/submission_result/expt15_b5_moreclass_4batch_160k")
    output_dir.mkdir(parents=True, exist_ok=True)

    orig_h = 9600
    orig_w = 14000
    patch_size = 640
    stride = 320
    num_classes = 8
    metadata_path = None

    metadata = load_metadata(metadata_path) if metadata_path else None

    predictions = load_predictions(pkl_path)

    segmentation_map = stitch_patches_from_pkl(
        predictions=predictions,
        orig_h=orig_h,
        orig_w=orig_w,
        patch_size=patch_size,
        stride=stride,
        metadata=metadata,
    )

    print("Final stitched map shape:", segmentation_map.shape)

    # # save the whole image to npy
    # np.save(output_dir / "segmentation_map.npy", segmentation_map)

    # # save as png
    # # Note: This is just the label image saved directly, not necessarily visually appealing
    # cv2.imwrite(str(output_dir / "segmentation_map.png"), segmentation_map)

    # Save binary masks for each class
    binary_masks = extract_binary_masks(segmentation_map, num_classes)
    np.save(output_dir / "binary_masks.npy", binary_masks)

    for cls_id in range(num_classes):
        mask = binary_masks[cls_id] * 255
        cv2.imwrite(str(output_dir / f"class_{cls_id}.png"), mask)

    # Generate submission CSV with RLE encoding
    masks_to_submission_rle(str(output_dir / "submission.csv"), binary_masks)

    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    main()