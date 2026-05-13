#!/usr/bin/env python3
# Run MMDetection Mask2Former on histmap patches and stitch instance label maps.

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from mmdet.apis import inference_detector, init_detector


def parse_args():
    parser = argparse.ArgumentParser(description="Infer patch instances and stitch full-image label maps.")
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--patch-index-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", default="/cluster/scratch/caizhi/vectorization/dataset", type=Path)
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-thr", default=0.5, type=float)
    parser.add_argument("--min-area", default=50, type=int)
    parser.add_argument("--valid-margin", default=128, type=int, help="Ignore this many pixels on overlapping patch borders.")
    parser.add_argument("--overlap-skip-ratio", default=0.5, type=float)
    parser.add_argument("--merge-touch-radius", default=2, type=int,
                        help="Merge a patch instance into an existing label if their masks touch within this radius.")
    parser.add_argument("--merge-min-contact", default=20, type=int,
                        help="Minimum touching/overlap pixels required for cross-patch label merge.")
    parser.add_argument("--max-labels", default=65000, type=int)
    parser.add_argument("--save-preview", action="store_true")
    return parser.parse_args()


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def group_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["sample_id"]].append(row)
    return grouped


def to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    if hasattr(value, "to_ndarray"):
        return value.to_ndarray()
    return np.asarray(value)


def get_pred_arrays(result):
    pred = result.pred_instances
    if len(pred) == 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64), np.empty((0, 1, 1), dtype=bool)
    scores = to_numpy(pred.scores).astype(np.float32)
    labels = to_numpy(pred.labels).astype(np.int64)
    masks = to_numpy(pred.masks).astype(bool)
    return scores, labels, masks


def valid_window(h, w, x, y, image_w, image_h, margin):
    if margin <= 0:
        return np.ones((h, w), dtype=bool)
    left = 0 if x == 0 else margin
    top = 0 if y == 0 else margin
    right = w if x + w >= image_w else max(left, w - margin)
    bottom = h if y + h >= image_h else max(top, h - margin)
    valid = np.zeros((h, w), dtype=bool)
    valid[top:bottom, left:right] = True
    return valid


def find_touch_merge_label(target, mask, radius, min_contact):
    if radius <= 0:
        return 0, 0
    if not np.any(target > 0):
        return 0, 0

    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    search = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    touching = search & (target > 0)
    if not np.any(touching):
        return 0, 0

    labels, counts = np.unique(target[touching], return_counts=True)
    keep = labels > 0
    labels = labels[keep]
    counts = counts[keep]
    if len(labels) == 0:
        return 0, 0

    best_index = int(np.argmax(counts))
    best_label = int(labels[best_index])
    best_contact = int(counts[best_index])
    if best_contact < min_contact:
        return 0, best_contact
    return best_label, best_contact


def load_roi(dataset_root, dataset_split, sample_id, shape):
    roi_path = dataset_root / dataset_split / f"{sample_id}-INPUT-MASK.png"
    if not roi_path.exists():
        return np.ones(shape, dtype=bool)
    roi = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
    if roi is None:
        raise FileNotFoundError(roi_path)
    if roi.shape != shape:
        raise ValueError(f"ROI size mismatch for {sample_id}: {roi.shape} vs {shape}")
    return roi > 0


def save_label_preview(label_map, path):
    rng = np.random.default_rng(17)
    labels = np.unique(label_map)
    preview = np.zeros(label_map.shape + (3,), dtype=np.uint8)
    for label in labels:
        if int(label) == 0:
            continue
        color = rng.integers(60, 255, size=3, dtype=np.uint8)
        preview[label_map == label] = color
    cv2.imwrite(str(path), preview)


def stitch_sample(model, sample_id, rows, args):
    image_h = max(int(row["image_height"]) for row in rows)
    image_w = max(int(row["image_width"]) for row in rows)
    label_map = np.zeros((image_h, image_w), dtype=np.uint32)
    roi = load_roi(args.dataset_root, args.dataset_split, sample_id, (image_h, image_w))
    next_label = 1
    accepted = 0
    skipped_overlap = 0
    skipped_area = 0
    merged_touch = 0

    for row in rows:
        patch_path = row.get("patch_path") or row["file_name"]
        x = int(row["x"])
        y = int(row["y"])
        w = int(row["width"])
        h = int(row["height"])
        valid = valid_window(h, w, x, y, image_w, image_h, args.valid_margin)

        result = inference_detector(model, patch_path)
        scores, labels, masks = get_pred_arrays(result)
        order = np.argsort(-scores)
        for idx in order:
            if scores[idx] < args.score_thr:
                continue
            if labels[idx] != 0:
                continue
            mask = masks[idx]
            if mask.shape[0] < h or mask.shape[1] < w:
                raise ValueError(f"Predicted mask {mask.shape} is smaller than patch {(h, w)}")
            mask = mask[:h, :w] & valid
            if int(mask.sum()) < args.min_area:
                skipped_area += 1
                continue

            full_roi = roi[y:y + h, x:x + w]
            mask &= full_roi
            if int(mask.sum()) < args.min_area:
                skipped_area += 1
                continue

            target = label_map[y:y + h, x:x + w]
            existing = target > 0
            area = int(mask.sum())
            if area == 0:
                skipped_area += 1
                continue

            merge_label, _ = find_touch_merge_label(
                target, mask, args.merge_touch_radius, args.merge_min_contact)
            assign = mask & ~existing
            if merge_label > 0:
                if int(assign.sum()) < args.min_area:
                    skipped_area += 1
                    continue
                target[assign] = merge_label
                accepted += 1
                merged_touch += 1
                continue

            overlap = int((mask & existing).sum())
            if overlap / float(area) > args.overlap_skip_ratio:
                skipped_overlap += 1
                continue
            if int(assign.sum()) < args.min_area:
                skipped_area += 1
                continue
            if next_label > args.max_labels:
                raise RuntimeError(f"Too many labels for {sample_id}; increase --max-labels or reduce predictions")
            target[assign] = next_label
            next_label += 1
            accepted += 1

    label_map[~roi] = 0
    return label_map, accepted, skipped_area, skipped_overlap, merged_touch


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.patch_index_csv)
    grouped = group_rows(rows)
    model = init_detector(args.config, args.checkpoint, device=args.device)

    summary_rows = []
    for sample_id in sorted(grouped):
        sample_dir = args.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        label_map, accepted, skipped_area, skipped_overlap, merged_touch = stitch_sample(model, sample_id, grouped[sample_id], args)
        np.save(sample_dir / "label_map.npy", label_map)
        cv2.imwrite(str(sample_dir / "label_map.tif"), label_map.astype(np.uint16))
        if args.save_preview:
            save_label_preview(label_map, sample_dir / "instance_preview.png")
        summary_rows.append({
            "sample_id": sample_id,
            "instances": int(label_map.max()),
            "accepted": accepted,
            "skipped_area": skipped_area,
            "skipped_overlap": skipped_overlap,
            "merged_touch": merged_touch,
        })
        print(f"{sample_id}: instances={int(label_map.max())} accepted={accepted} skipped_area={skipped_area} skipped_overlap={skipped_overlap} merged_touch={merged_touch}")

    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "instances", "accepted", "skipped_area", "skipped_overlap", "merged_touch"])
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
