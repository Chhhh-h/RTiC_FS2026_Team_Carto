from pathlib import Path
import numpy as np
import cv2
from pycocotools import mask as maskUtils


def load_binary_mask(mask_path: Path):
    """
    Read a 0/255 mask png and convert it to {0,1}.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")
    return (mask > 127).astype(np.uint8)


def load_all_class_masks(mask_dir: Path, num_classes=8):
    """
    Load class_0.png ... class_{num_classes-1}.png into a (C,H,W) array.
    """
    masks = []
    for cls_id in range(num_classes):
        mask_path = mask_dir / f"class_{cls_id}.png"
        masks.append(load_binary_mask(mask_path))
    masks = np.stack(masks, axis=0)  # (C,H,W)
    return masks


def save_debug_masks(mask_array, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for cls_id in range(mask_array.shape[0]):
        cv2.imwrite(
            str(out_dir / f"class_{cls_id}.png"),
            (mask_array[cls_id] * 255).astype(np.uint8)
        )


def mask_to_rle(binary_mask):
    """
    Encode a binary mask as COCO compressed RLE string.
    """
    binary_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = maskUtils.encode(binary_mask)
    return rle["counts"].decode("utf-8")


def mask_to_submission_strings_rle(mask: np.ndarray, skip_class_0=True):
    """
    mask: (C,H,W), each channel is binary.
    Output rows: 'ID,rle'
    """
    for cls_id, binary in enumerate(mask):
        if skip_class_0 and cls_id == 0:
            continue
        rle = mask_to_rle(binary)
        yield f"{cls_id},{rle}"


def masks_to_submission_rle(submission_filename, mask: np.ndarray, skip_class_0=True):
    with open(submission_filename, "w", encoding="utf-8") as f:
        f.write("ID,rle\n")
        for s in mask_to_submission_strings_rle(mask, skip_class_0=skip_class_0):
            f.write(s + "\n")


def fuse_convnext_5_7_into_segformer(
    seg_masks: np.ndarray,
    conv_masks: np.ndarray,
    replace_classes=(5, 7),
    priority=(5, 7),
):
    """
    Strict replacement:
    - Wherever convnext predicts class 5 or 7, remove all segformer classes there
    - Then write convnext class 5/7 back

    Args:
        seg_masks:  (C,H,W) from segformer
        conv_masks: (C,H,W) from convnext
        replace_classes: classes to replace, default (5,7)
        priority: overlap resolution order inside convnext replace classes.
                  Later class overwrites earlier class logically.
                  Example:
                    priority=(5,7) means 7 has higher priority than 5
                    priority=(7,5) means 5 has higher priority than 7
    """
    out_masks = seg_masks.copy()
    c, h, w = out_masks.shape

    # 1) Build replacement union from convnext class 5 and 7
    replace_union = np.zeros((h, w), dtype=np.uint8)
    for cls_id in replace_classes:
        replace_union |= conv_masks[cls_id]

    # 2) Remove ALL segformer classes on those pixels
    out_masks[:, replace_union == 1] = 0

    # 3) Resolve potential overlap between convnext class 5 and 7
    assigned = np.zeros((h, w), dtype=np.uint8)
    for cls_id in priority:
        current = conv_masks[cls_id].copy()
        current[assigned == 1] = 0
        out_masks[cls_id] = current
        assigned |= current

    return out_masks


def check_overlap(mask_array, name="mask"):
    """
    Check if one pixel belongs to multiple classes.
    """
    overlap_count = mask_array.sum(axis=0)
    num_overlap = int((overlap_count > 1).sum())
    num_fg = int((overlap_count > 0).sum())
    print(f"[{name}] foreground pixels: {num_fg}")
    print(f"[{name}] overlap pixels (>1 class): {num_overlap}")
    return num_overlap


def main():
    # =========================
    # 修改这里
    # =========================
    segformer_dir = Path("/cluster/scratch/pangyi/reto/submission_result/expt14_b5_class_cedice_160k_overlap")
    convnext_dir = Path("/cluster/scratch/pangyi/reto/submission_result/convnext_final")
    output_dir = Path("/cluster/scratch/pangyi/reto/submission_result/fuse_convenext_segformer")
    output_dir.mkdir(parents=True, exist_ok=True)

    num_classes = 8

    # 是否跳过 class 0 写入 submission
    # 你原来的代码是跳过 class 0 的，所以默认 True
    skip_class_0 = True

    # 用 convnext 替换 segformer 的哪些类
    replace_classes = (5, 7)

    # 处理 convnext 的 5 和 7 如果互相重叠时的优先级
    # (5,7) 表示 7 优先
    priority = (5, 7)

    # =========================
    # 读 mask
    # =========================
    seg_masks = load_all_class_masks(segformer_dir, num_classes=num_classes)
    conv_masks = load_all_class_masks(convnext_dir, num_classes=num_classes)

    print("Loaded segformer masks:", seg_masks.shape)
    print("Loaded convnext masks:", conv_masks.shape)

    # 基本检查
    if seg_masks.shape != conv_masks.shape:
        raise ValueError(
            f"Shape mismatch: segformer {seg_masks.shape}, convnext {conv_masks.shape}"
        )

    check_overlap(seg_masks, name="segformer_before")
    check_overlap(conv_masks, name="convnext_before")

    # =========================
    # 融合
    # =========================
    fused_masks = fuse_convnext_5_7_into_segformer(
        seg_masks=seg_masks,
        conv_masks=conv_masks,
        replace_classes=replace_classes,
        priority=priority,
    )

    check_overlap(fused_masks, name="fused_after")

    # 保存中间结果，方便你检查
    save_debug_masks(fused_masks, output_dir / "fused_masks_png")

    # 生成 submission csv
    submission_path = output_dir / "submission_fused.csv"
    masks_to_submission_rle(
        str(submission_path),
        fused_masks,
        skip_class_0=skip_class_0
    )

    print(f"Saved fused submission to: {submission_path}")


if __name__ == "__main__":
    main()