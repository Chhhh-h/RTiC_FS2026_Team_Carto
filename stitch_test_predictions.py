#!/usr/bin/env python3

import argparse
import csv
import os
import pickle

import numpy as np
from PIL import Image


def load_predictions(path: str):
    with open(path, "rb") as f:
        preds = pickle.load(f)
    if not isinstance(preds, list):
        raise TypeError(f"Expected a list from {path}, but got {type(preds).__name__}")
    return preds


def load_patch_index(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in patch index: {path}")
    return rows


def sort_rows_like_mmseg_dataset(rows):
    # mmseg loads images from directory using lexicographic filename order.
    # Align patch-index rows to that same order before zipping with predictions.
    def sort_key(row):
        image_path = row.get("image_path", "")
        if image_path:
            return os.path.basename(image_path)
        patch_name = row.get("patch_name", "")
        return f"{patch_name}.png" if patch_name else ""

    return sorted(rows, key=sort_key)


def infer_canvas_size(rows):
    height = max(int(row["y"]) + int(row["height"]) for row in rows)
    width = max(int(row["x"]) + int(row["width"]) for row in rows)
    map_ids = sorted({row["map_id"] for row in rows})
    if len(map_ids) != 1:
        raise ValueError(f"Expected exactly one map_id, but got {map_ids}")
    return map_ids[0], height, width


def group_by_map_id(preds, rows):
    grouped = {}
    for pred, row in zip(preds, rows):
        map_id = row["map_id"]
        grouped.setdefault(map_id, {"preds": [], "rows": []})
        grouped[map_id]["preds"].append(pred)
        grouped[map_id]["rows"].append(row)
    return grouped


def label_map_to_binary_channels(
        label_map: np.ndarray,
        num_classes: int,
        zero_based_labels: bool = False) -> np.ndarray:
    if label_map.ndim != 2:
        raise ValueError(f"Expected label map [H,W], but got shape {label_map.shape}")

    channels = np.zeros((num_classes, label_map.shape[0], label_map.shape[1]), dtype=np.float32)
    if zero_based_labels:
        # label ids are 0..num_classes-1
        for cls_id in range(num_classes):
            channels[cls_id] = (label_map == cls_id).astype(np.uint8)
    else:
        # label ids are 1..num_classes, with 0 reserved for background
        for cls_id in range(1, num_classes + 1):
            channels[cls_id - 1] = (label_map == cls_id).astype(np.uint8)
    return channels


def score_map_to_binary_channels(score_map: np.ndarray, num_classes: int, threshold: float) -> np.ndarray:
    if score_map.ndim != 3:
        raise ValueError(f"Expected score map [C,H,W], but got shape {score_map.shape}")
    if score_map.shape[0] != num_classes:
        raise ValueError(
            f"Expected {num_classes} channels in score map, but got shape {score_map.shape}"
        )
    return (score_map > threshold).astype(np.float32)


def prediction_to_binary_channels(
        pred,
        num_classes: int,
        threshold: float,
        zero_based_labels: bool = False) -> np.ndarray:
    pred = np.asarray(pred)
    if pred.ndim == 2:
        return label_map_to_binary_channels(pred, num_classes, zero_based_labels=zero_based_labels)
    if pred.ndim == 3:
        return score_map_to_binary_channels(pred, num_classes, threshold)
    raise ValueError(f"Unsupported prediction shape: {pred.shape}")


def parse_class_names(class_names_arg: str, num_classes: int):
    if not class_names_arg:
        return [f"class_{idx + 1}" for idx in range(num_classes)]

    class_names = [name.strip() for name in class_names_arg.split(",") if name.strip()]
    if len(class_names) != num_classes:
        raise ValueError(
            f"Expected {num_classes} class names, but got {len(class_names)} from: {class_names_arg}"
        )
    return class_names


def save_class_pngs(stitched: np.ndarray, output_dir: str, class_names):
    os.makedirs(output_dir, exist_ok=True)
    for cls_idx, class_name in enumerate(class_names):
        png_path = os.path.join(output_dir, f"{cls_idx + 1:02d}_{class_name}.png")
        Image.fromarray((stitched[cls_idx] * 255).astype(np.uint8), mode="L").save(png_path)
        print(f"Saved class PNG: {png_path}")


def stitch_one_map(
        preds,
        rows,
        num_classes: int,
        threshold: float,
        zero_based_labels: bool) -> np.ndarray:
    _, canvas_h, canvas_w = infer_canvas_size(rows)
    stitched_sum = np.zeros((num_classes, canvas_h, canvas_w), dtype=np.float32)
    stitched_count = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    for pred, row in zip(preds, rows):
        y = int(row["y"])
        x = int(row["x"])
        h = int(row["height"])
        w = int(row["width"])

        patch_mask = prediction_to_binary_channels(
            pred,
            num_classes,
            threshold,
            zero_based_labels=zero_based_labels)

        if patch_mask.shape[1] < h or patch_mask.shape[2] < w:
            raise ValueError(
                f"Prediction patch is smaller than target crop: pred {patch_mask.shape}, target {(h, w)}"
            )

        patch_crop = patch_mask[:, :h, :w]
        stitched_sum[:, y : y + h, x : x + w] += patch_crop
        stitched_count[y : y + h, x : x + w] += 1.0

    if np.any(stitched_count == 0):
        raise ValueError("Some canvas pixels were not covered by any patch.")

    stitched = stitched_sum / stitched_count[None, :, :]
    return (stitched >= 0.5).astype(np.uint8)


def output_path_for_map(output_npy: str, map_id: str, num_maps: int) -> str:
    if num_maps == 1:
        return output_npy
    out_dir = os.path.dirname(output_npy) or "."
    stem = os.path.splitext(os.path.basename(output_npy))[0]
    if stem.endswith("_map3"):
        stem = stem[:-5]
    return os.path.join(out_dir, f"{map_id}_{stem}.npy")


def main():
    parser = argparse.ArgumentParser(
        description="Stitch patch-level mmseg predictions into a [C,H,W] mask for submission"
    )
    parser.add_argument(
        "--input-pkl",
        type=str,
        required=True,
        help="Path to tools/test.py output .pkl",
    )
    parser.add_argument(
        "--patch-index-csv",
        type=str,
        default="data/patches_640_split/patch_index_test.csv",
        help="CSV with patch_name/map_id/y/x/height/width for test patches",
    )
    parser.add_argument(
        "--output-npy",
        type=str,
        required=True,
        help="Output stitched .npy path, shape [C,H,W]",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=7,
        help="Number of foreground classes expected by submission",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used only when predictions are score maps [C,H,W]",
    )
    parser.add_argument(
        "--zero-based-labels",
        action="store_true",
        help="Use this when 2D predictions use class ids 0..C-1 rather than 1..C",
    )
    parser.add_argument(
        "--class-png-dir",
        type=str,
        default=None,
        help="Optional directory to save one binary PNG per class after stitching",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        default="",
        help="Optional comma-separated class names used for naming PNG files",
    )
    args = parser.parse_args()

    preds = load_predictions(args.input_pkl)
    rows = sort_rows_like_mmseg_dataset(load_patch_index(args.patch_index_csv))

    if len(preds) != len(rows):
        raise ValueError(
            f"Prediction count ({len(preds)}) does not match patch index rows ({len(rows)})"
        )

    grouped = group_by_map_id(preds, rows)
    num_maps = len(grouped)
    os.makedirs(os.path.dirname(args.output_npy) or ".", exist_ok=True)

    class_names = parse_class_names(args.class_names, args.num_classes) if args.class_png_dir else None

    for map_id in sorted(grouped):
        item = grouped[map_id]
        stitched = stitch_one_map(
            item["preds"],
            item["rows"],
            args.num_classes,
            args.threshold,
            zero_based_labels=args.zero_based_labels)

        out_path = output_path_for_map(args.output_npy, map_id, num_maps)
        np.save(out_path, stitched)

        if args.class_png_dir:
            png_dir = os.path.join(args.class_png_dir, map_id) if num_maps > 1 else args.class_png_dir
            save_class_pngs(stitched, png_dir, class_names)

        print(f"Map id: {map_id}")
        print(f"Saved stitched mask to: {out_path}")
        print(f"Stitched shape: {stitched.shape}")


if __name__ == "__main__":
    main()
