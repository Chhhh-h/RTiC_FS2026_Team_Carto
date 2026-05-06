import argparse
import csv
import os
import re

import numpy as np
import tifffile
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


def build_start_points(length: int, patch_size: int, stride: int):
    if length <= patch_size:
        return [0]

    points = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if points[-1] != last:
        points.append(last)
    return points


def ensure_hwc_7(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 3:
        raise ValueError(f"Mask must be 3D, but got shape {mask.shape}")

    if mask.shape[-1] == 7:
        return mask
    if mask.shape[0] == 7:
        return np.transpose(mask, (1, 2, 0))

    raise ValueError(f"Mask must have 7 channels, but got shape {mask.shape}")


def save_rgb_patch(rgb_patch: np.ndarray, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(rgb_patch.astype(np.uint8)).save(out_path)


def save_seg_label_tif(mask_label: np.ndarray, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tifffile.imwrite(out_path, mask_label.astype(np.uint8))


def pad_patch_if_needed(arr: np.ndarray, patch_size: int):
    h, w = arr.shape[:2]
    if h == patch_size and w == patch_size:
        return arr

    if arr.ndim == 3:
        padded = np.zeros((patch_size, patch_size, arr.shape[2]), dtype=arr.dtype)
        padded[:h, :w, :] = arr
    else:
        padded = np.zeros((patch_size, patch_size), dtype=arr.dtype)
        padded[:h, :w] = arr
    return padded


def write_csv(csv_path, rows, fieldnames):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def multi_channel_mask_to_label_map(mask: np.ndarray) -> np.ndarray:
    """Convert a competition H,W,7 mask into a single-channel label map.

    Output labels:
      0: background
      1..7: competition classes in channel order
    """
    mask = np.asarray(mask)
    if mask.ndim != 3 or mask.shape[-1] != 7:
        raise ValueError(f"Expected mask shape H,W,7 but got {mask.shape}")

    mask = (mask > 0).astype(np.uint8)
    label = np.zeros(mask.shape[:2], dtype=np.uint8)
    active = mask.any(axis=2)
    if np.any(active):
        label[active] = np.argmax(mask[active], axis=1).astype(np.uint8) + 1
    return label


def build_segformer_layout(base_dir: str):
    paths = {
        "train_comp_img_dir": os.path.join(base_dir, "images", "training_competition"),
        "val_img_dir": os.path.join(base_dir, "images", "validation"),
        "test_img_dir": os.path.join(base_dir, "images", "test"),
        "train_comp_ann_dir": os.path.join(base_dir, "annotations", "training_competition_background"),
        "val_ann_dir": os.path.join(base_dir, "annotations", "validation_background"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def split_competition_patch(map_id: str, x: int, y: int, val_ratio: float):
    if val_ratio <= 0:
        return "training"
    if val_ratio >= 1:
        return "validation"

    bucket = ((x // 32) * 73856093 + (y // 32) * 19349663 + sum(ord(c) for c in map_id)) % 100
    return "validation" if bucket < int(round(val_ratio * 100)) else "training"


def build_row(
        patch_name,
        source,
        map_id,
        split,
        y,
        x,
        height,
        width,
        image_path,
        mask_path,
        label_map):
    class_pixels = [(label_map == cls_id).sum() for cls_id in range(1, 8)]
    return {
        "patch_name": patch_name,
        "source": source,
        "map_id": map_id,
        "split": split,
        "y": y,
        "x": x,
        "height": height,
        "width": width,
        "image_path": image_path,
        "mask_path": mask_path,
        "is_empty": int(np.all(label_map == 0)),
        "class_0_pixels": int(class_pixels[0]),
        "class_1_pixels": int(class_pixels[1]),
        "class_2_pixels": int(class_pixels[2]),
        "class_3_pixels": int(class_pixels[3]),
        "class_4_pixels": int(class_pixels[4]),
        "class_5_pixels": int(class_pixels[5]),
        "class_6_pixels": int(class_pixels[6]),
    }


def cut_train_map(rgb_path, mask_path, map_id, layout, patch_size, stride, save_empty, val_ratio):
    print(f"\nLoading competition train {map_id} ...")
    rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    mask = tifffile.imread(mask_path)
    mask = ensure_hwc_7(mask)
    mask = (mask > 0).astype(np.uint8)

    if rgb.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Image-mask size mismatch: rgb={rgb.shape}, mask={mask.shape}")

    h, w = rgb.shape[:2]
    ys = build_start_points(h, patch_size, stride)
    xs = build_start_points(w, patch_size, stride)

    rows = []
    total_saved = 0
    total_skipped_empty = 0

    pbar = tqdm(total=len(ys) * len(xs), desc=f"Cut competition train {map_id}")
    for y in ys:
        for x in xs:
            rgb_patch = rgb[y:y + patch_size, x:x + patch_size, :]
            mask_patch = mask[y:y + patch_size, x:x + patch_size, :]

            rgb_patch = pad_patch_if_needed(rgb_patch, patch_size)
            mask_patch = pad_patch_if_needed(mask_patch, patch_size)

            label_map = multi_channel_mask_to_label_map(mask_patch)
            if np.all(label_map == 0) and not save_empty:
                total_skipped_empty += 1
                pbar.update(1)
                continue

            patch_name = f"{map_id}_y{y}_x{x}"
            split = split_competition_patch(map_id=map_id, x=x, y=y, val_ratio=val_ratio)
            if split == "training":
                image_abs = os.path.join(layout["train_comp_img_dir"], patch_name + ".png")
                mask_abs = os.path.join(layout["train_comp_ann_dir"], patch_name + ".tif")
                image_rel = os.path.join("images", "training_competition", patch_name + ".png")
                mask_rel = os.path.join("annotations", "training_competition_background", patch_name + ".tif")
            else:
                image_abs = os.path.join(layout["val_img_dir"], patch_name + ".png")
                mask_abs = os.path.join(layout["val_ann_dir"], patch_name + ".tif")
                image_rel = os.path.join("images", "validation", patch_name + ".png")
                mask_rel = os.path.join("annotations", "validation_background", patch_name + ".tif")

            save_rgb_patch(rgb_patch, image_abs)
            save_seg_label_tif(label_map, mask_abs)

            rows.append(build_row(
                patch_name=patch_name,
                source="competition",
                map_id=map_id,
                split=split,
                y=y,
                x=x,
                height=patch_size,
                width=patch_size,
                image_path=image_rel,
                mask_path=mask_rel,
                label_map=label_map,
            ))

            total_saved += 1
            pbar.update(1)

    pbar.close()
    print(f"{map_id}: saved={total_saved}, skipped_empty={total_skipped_empty}")
    return rows


def cut_test_map(rgb_path, map_id, out_root, patch_size, stride):
    print(f"\nLoading test {map_id} ...")
    rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)

    h, w = rgb.shape[:2]
    ys = build_start_points(h, patch_size, stride)
    xs = build_start_points(w, patch_size, stride)

    rows = []
    total_saved = 0

    pbar = tqdm(total=len(ys) * len(xs), desc=f"Cut test {map_id}")
    for y in ys:
        for x in xs:
            rgb_patch = rgb[y:y + patch_size, x:x + patch_size, :]
            rgb_patch = pad_patch_if_needed(rgb_patch, patch_size)

            patch_name = f"{map_id}_y{y}_x{x}"
            image_rel = os.path.join("images", "test", patch_name + ".png")
            image_abs = os.path.join(out_root, image_rel)

            save_rgb_patch(rgb_patch, image_abs)

            rows.append({
                "patch_name": patch_name,
                "source": "competition_test",
                "map_id": map_id,
                "y": y,
                "x": x,
                "height": patch_size,
                "width": patch_size,
                "image_path": image_rel,
            })

            total_saved += 1
            pbar.update(1)

    pbar.close()
    print(f"{map_id}: saved={total_saved}")
    return rows


def find_test_images(test_root):
    test_images = []
    for fname in sorted(os.listdir(test_root)):
        match = re.match(r"^rgb_(map\d+)\.png$", fname)
        if match:
            test_images.append((fname, match.group(1)))
    if not test_images:
        raise FileNotFoundError(f"No rgb_map*.png test images found in: {test_root}")
    return test_images


def main():
    parser = argparse.ArgumentParser(
        description="Cut original competition train/test maps into SegFormer-ready patches."
    )
    parser.add_argument("--train-root", type=str, default="data_competition_original/train")
    parser.add_argument("--test-root", type=str, default="data_competition_original/test_final")
    parser.add_argument("--output-root", type=str, default="data")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--save-empty-train", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    segformer_root = os.path.join(args.output_root, f"patches_{args.patch_size}_split")
    layout = build_segformer_layout(segformer_root)

    train_pairs = [
        ("rgb_map1.png", "gt_mask_map1.tif", "map1"),
        ("rgb_map2.png", "gt_mask_map2.tif", "map2"),
    ]

    competition_rows = []
    for rgb_name, mask_name, map_id in train_pairs:
        rgb_path = os.path.join(args.train_root, rgb_name)
        mask_path = os.path.join(args.train_root, mask_name)

        if not os.path.exists(rgb_path):
            raise FileNotFoundError(f"Train RGB not found: {rgb_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Train mask not found: {mask_path}")

        rows = cut_train_map(
            rgb_path=rgb_path,
            mask_path=mask_path,
            map_id=map_id,
            layout=layout,
            patch_size=args.patch_size,
            stride=args.stride,
            save_empty=args.save_empty_train,
            val_ratio=args.val_ratio,
        )
        competition_rows.extend(rows)

    train_fieldnames = [
        "patch_name",
        "source",
        "map_id",
        "split",
        "y",
        "x",
        "height",
        "width",
        "image_path",
        "mask_path",
        "is_empty",
        "class_0_pixels",
        "class_1_pixels",
        "class_2_pixels",
        "class_3_pixels",
        "class_4_pixels",
        "class_5_pixels",
        "class_6_pixels",
    ]
    write_csv(os.path.join(segformer_root, "patch_index_all.csv"), competition_rows, train_fieldnames)
    write_csv(os.path.join(segformer_root, "patch_index_competition.csv"), competition_rows, train_fieldnames)

    all_test_rows = []
    for rgb_name, map_id in find_test_images(args.test_root):
        rows = cut_test_map(
            rgb_path=os.path.join(args.test_root, rgb_name),
            map_id=map_id,
            out_root=segformer_root,
            patch_size=args.patch_size,
            stride=args.stride,
        )
        all_test_rows.extend(rows)

    write_csv(
        os.path.join(segformer_root, "patch_index_test.csv"),
        all_test_rows,
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

    print("\nAll done.")
    print(f"SegFormer-ready data saved to: {segformer_root}")
    print(f"Competition train images: {layout['train_comp_img_dir']}")
    print(f"Competition train labels: {layout['train_comp_ann_dir']}")
    print(f"Validation images: {layout['val_img_dir']}")
    print(f"Validation labels: {layout['val_ann_dir']}")
    print(f"Test images: {layout['test_img_dir']}")


if __name__ == "__main__":
    main()
