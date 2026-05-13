#!/usr/bin/env python3
# Build pseudo-instance COCO annotations from full historical-map binary masks.

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils


CATEGORY = {"id": 1, "name": "building_block", "supercategory": "building"}


def parse_args():
    parser = argparse.ArgumentParser(description="Create patch COCO instance data from histmap GT masks.")
    parser.add_argument("--dataset-root", default="/cluster/scratch/caizhi/vectorization/dataset", type=Path)
    parser.add_argument("--output-root", default="data/histmap_instance", type=Path)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--patch-size", default=1024, type=int)
    parser.add_argument("--stride", default=768, type=int)
    parser.add_argument("--gt-threshold", default=0, type=int)
    parser.add_argument("--min-full-area", default=25, type=int)
    parser.add_argument("--min-visible-area", default=25, type=int)
    parser.add_argument("--jpeg-quality", default=95, type=int)
    parser.add_argument("--keep-empty-train", action="store_true")
    parser.add_argument("--drop-empty-val", action="store_true")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def read_image(path, flags):
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(path)
    return image


def iter_sample_ids(split_dir):
    for image_path in sorted(split_dir.glob("*-INPUT.jpg")):
        yield image_path.name.split("-INPUT.jpg")[0]


def iter_windows(width, height, patch_size, stride):
    xs = list(range(0, max(width - patch_size, 0) + 1, stride)) or [0]
    ys = list(range(0, max(height - patch_size, 0) + 1, stride)) or [0]
    if xs[-1] != max(width - patch_size, 0):
        xs.append(max(width - patch_size, 0))
    if ys[-1] != max(height - patch_size, 0):
        ys.append(max(height - patch_size, 0))
    for y in ys:
        for x in xs:
            yield x, y, min(patch_size, width - x), min(patch_size, height - y)


def encode_binary_mask(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def bbox_intersects(ax, ay, aw, ah, bx, by, bw, bh):
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


def make_component_table(gt, roi, gt_threshold, min_full_area):
    binary = (gt > gt_threshold).astype(np.uint8)
    if roi is not None:
        binary[roi <= 0] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    component_ids = []
    for cid in range(1, count):
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area >= min_full_area:
            component_ids.append(cid)
    return labels, stats, component_ids


def build_patch_annotations(labels, stats, component_ids, x, y, w, h, min_visible_area, image_id, start_ann_id):
    annotations = []
    ann_id = start_ann_id
    for cid in component_ids:
        bx = int(stats[cid, cv2.CC_STAT_LEFT])
        by = int(stats[cid, cv2.CC_STAT_TOP])
        bw = int(stats[cid, cv2.CC_STAT_WIDTH])
        bh = int(stats[cid, cv2.CC_STAT_HEIGHT])
        if not bbox_intersects(x, y, w, h, bx, by, bw, bh):
            continue

        ox0 = max(x, bx)
        oy0 = max(y, by)
        ox1 = min(x + w, bx + bw)
        oy1 = min(y + h, by + bh)
        local = labels[oy0:oy1, ox0:ox1] == cid
        visible_area = int(local.sum())
        if visible_area < min_visible_area:
            continue

        patch_mask = np.zeros((h, w), dtype=np.uint8)
        patch_mask[oy0 - y:oy1 - y, ox0 - x:ox1 - x] = local.astype(np.uint8)
        rle = encode_binary_mask(patch_mask)
        area = float(mask_utils.area(rle))
        if area < min_visible_area:
            continue
        bbox = [float(v) for v in mask_utils.toBbox(rle)]
        annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": 1,
            "segmentation": rle,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0,
        })
        ann_id += 1
    return annotations, ann_id


def prepare_split(args, source_split, out_split, keep_empty, image_id_start, ann_id_start):
    split_dir = args.dataset_root / source_split
    image_dir = args.output_root / "images" / out_split
    image_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    patch_rows = []
    image_id = image_id_start
    ann_id = ann_id_start

    for sample_id in iter_sample_ids(split_dir):
        image_path = split_dir / f"{sample_id}-INPUT.jpg"
        gt_path = split_dir / f"{sample_id}-OUTPUT-GT.png"
        roi_path = split_dir / f"{sample_id}-INPUT-MASK.png"
        if not gt_path.exists():
            raise FileNotFoundError(gt_path)

        image = read_image(image_path, cv2.IMREAD_COLOR)
        gt = read_image(gt_path, cv2.IMREAD_GRAYSCALE)
        roi = read_image(roi_path, cv2.IMREAD_GRAYSCALE) if roi_path.exists() else None
        if gt.shape[:2] != image.shape[:2]:
            raise ValueError(f"Image/GT size mismatch for {sample_id}: {image.shape[:2]} vs {gt.shape[:2]}")
        if roi is not None and roi.shape[:2] != image.shape[:2]:
            raise ValueError(f"Image/ROI size mismatch for {sample_id}: {image.shape[:2]} vs {roi.shape[:2]}")

        labels, stats, component_ids = make_component_table(gt, roi, args.gt_threshold, args.min_full_area)
        height, width = image.shape[:2]
        for x, y, w, h in iter_windows(width, height, args.patch_size, args.stride):
            file_name = f"{sample_id}__x{x:05d}_y{y:05d}.jpg"
            patch_anns, next_ann_id = build_patch_annotations(
                labels, stats, component_ids, x, y, w, h, args.min_visible_area, image_id, ann_id)
            if not patch_anns and not keep_empty:
                continue

            patch = image[y:y + h, x:x + w]
            cv2.imwrite(str(image_dir / file_name), patch, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
            images.append({
                "id": image_id,
                "file_name": file_name,
                "width": int(w),
                "height": int(h),
            })
            annotations.extend(patch_anns)
            patch_rows.append({
                "split": out_split,
                "sample_id": sample_id,
                "file_name": file_name,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "image_width": width,
                "image_height": height,
                "num_instances": len(patch_anns),
            })
            ann_id = next_ann_id
            image_id += 1

    return images, annotations, patch_rows, image_id, ann_id


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def write_patch_index(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "sample_id", "file_name", "x", "y", "width", "height", "image_width", "image_height", "num_instances"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.clean and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    train_images, train_anns, train_rows, next_image_id, next_ann_id = prepare_split(
        args, args.train_split, "train", args.keep_empty_train, 1, 1)
    val_images, val_anns, val_rows, _, _ = prepare_split(
        args, args.val_split, "val", not args.drop_empty_val, next_image_id, next_ann_id)

    ann_dir = args.output_root / "annotations"
    write_json(ann_dir / "instances_train.json", {"images": train_images, "annotations": train_anns, "categories": [CATEGORY]})
    write_json(ann_dir / "instances_val.json", {"images": val_images, "annotations": val_anns, "categories": [CATEGORY]})
    write_patch_index(args.output_root / "patch_index_train.csv", train_rows)
    write_patch_index(args.output_root / "patch_index_val.csv", val_rows)

    print(f"train patches={len(train_images)} annotations={len(train_anns)}")
    print(f"val patches={len(val_images)} annotations={len(val_anns)}")
    print(f"saved to {args.output_root}")


if __name__ == "__main__":
    main()
