import json
from pathlib import Path

import cv2
import numpy as np


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


def pad_image_to_multiple(image: np.ndarray, patch_size: int = 640, pad_value=0):
    """
    Resize the image to a size that is an integer multiple of patch_size.

    Args:
        image: input image, shape (H, W) or (H, W, C)
        patch_size: patch size
        pad_value: padding value, default 0

    Returns:
        padded_image: padded image
        orig_h, orig_w: original height and width
        pad_h, pad_w: padded height and width
    """
    orig_h, orig_w = image.shape[:2]
    pad_h = ((orig_h + patch_size - 1) // patch_size) * patch_size
    pad_w = ((orig_w + patch_size - 1) // patch_size) * patch_size

    if image.ndim == 2:
        padded_image = np.full((pad_h, pad_w), pad_value, dtype=image.dtype)
        padded_image[:orig_h, :orig_w] = image
    else:
        channels = image.shape[2]
        if isinstance(pad_value, (list, tuple)):
            fill_value = np.array(pad_value, dtype=image.dtype)
            padded_image = np.zeros((pad_h, pad_w, channels), dtype=image.dtype)
            padded_image[...] = fill_value
        else:
            padded_image = np.full((pad_h, pad_w, channels), pad_value, dtype=image.dtype)
        padded_image[:orig_h, :orig_w, :] = image

    return padded_image, orig_h, orig_w, pad_h, pad_w


def split_and_save_patches(
    image_path: str,
    output_dir: str,
    patch_size: int = 640,
    stride: int = 640,
    pad_value=0,
    image_read_flag=cv2.IMREAD_UNCHANGED,
):
    """
    Split an image into patches of size patch_size x patch_size, padding the edges if necessary, and save them.

    Args:
        image_path: Path to the input image
        output_dir: Output directory
        patch_size: Size of each patch
        stride: Sliding step between adjacent patches
        pad_value: Padding value for edges
        image_read_flag: cv2 reading flag
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path), image_read_flag)
    if image is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")

    padded_image, orig_h, orig_w, pad_h, pad_w = pad_image_to_multiple(
        image, patch_size=patch_size, pad_value=pad_value
    )

    y_starts = compute_start_positions(pad_h, patch_size, stride)
    x_starts = compute_start_positions(pad_w, patch_size, stride)
    num_rows = len(y_starts)
    num_cols = len(x_starts)

    patch_records = []
    patch_idx = 0

    stem = image_path.stem
    suffix = image_path.suffix if image_path.suffix else ".png"

    for row, y in enumerate(y_starts):
        for col, x in enumerate(x_starts):
            patch = padded_image[y:y + patch_size, x:x + patch_size]

            patch_name = f"{stem}_r{row:03d}_c{col:03d}{suffix}"
            patch_path = output_dir / patch_name

            success = cv2.imwrite(str(patch_path), patch)
            if not success:
                raise RuntimeError(f"Failed to save patch: {patch_path}")

            patch_records.append({
                "patch_index": patch_idx,
                "patch_name": patch_name,
                "row": row,
                "col": col,
                "y_start": y,
                "x_start": x,
                "y_end": y + patch_size,
                "x_end": x + patch_size,
            })
            patch_idx += 1

    metadata = {
        "original_image": str(image_path),
        "original_height": orig_h,
        "original_width": orig_w,
        "padded_height": pad_h,
        "padded_width": pad_w,
        "patch_size": patch_size,
        "stride": stride,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "num_patches": len(patch_records),
        "patches": patch_records,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"original shape: ({orig_h}, {orig_w})")
    print(f"padded shape: ({pad_h}, {pad_w})")
    print(f"patch grid: {num_rows} x {num_cols}")
    print(f"patch count: {len(patch_records)}")
    print(f"patches saved to: {output_dir}")
    print(f"metadata saved to: {metadata_path}")




def main():
    image_path = "/cluster/scratch/pangyi/reto/data/origin/test/test/rgb_map3.png"
    output_dir = "/cluster/scratch/pangyi/reto/data/test_patches_640_overlap320"
    patch_size = 640
    stride = 320
    pad_value = 0

    split_and_save_patches(
        image_path=image_path,
        output_dir=output_dir,
        patch_size=patch_size,
        stride=stride,
        pad_value=pad_value,
    )


if __name__ == "__main__":
    main()

