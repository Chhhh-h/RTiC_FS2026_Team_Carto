import os
import argparse
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model.unet.unet import UNET
from data import TestPatchFolderDataset
from mask_to_submission import masks_to_submission_rle

Image.MAX_IMAGE_PIXELS = None
NUM_CLASSES = 8


def save_label_map(label_map: np.ndarray, path: str):
    Image.fromarray(label_map.astype(np.uint8)).save(path)


def save_color_map(label_map: np.ndarray, path: str):
    palette = np.array([
        [0, 0, 0],         # 0 background
        [0, 0, 255],       # 1
        [0, 128, 0],       # 2
        [0, 255, 255],     # 3
        [128, 0, 128],     # 4
        [128, 255, 255],   # 5
        [255, 0, 0],       # 6
        [128, 128, 128],   # 7
    ], dtype=np.uint8)

    color = palette[label_map]
    Image.fromarray(color).save(path)


def build_weight_map(patch_size: int):
    yy, xx = np.mgrid[0:patch_size, 0:patch_size]
    cy = (patch_size - 1) / 2.0
    cx = (patch_size - 1) / 2.0
    sigma = patch_size / 4.0

    weight = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma * sigma))
    weight = weight.astype(np.float32)
    weight = weight / np.max(weight)
    return weight


def predict_with_tta(model, images, use_tta=False):
    logits_sum = model(images)

    if not use_tta:
        return logits_sum

    # horizontal flip
    images_h = torch.flip(images, dims=[3])
    logits_h = model(images_h)
    logits_h = torch.flip(logits_h, dims=[3])
    logits_sum += logits_h

    # vertical flip
    images_v = torch.flip(images, dims=[2])
    logits_v = model(images_v)
    logits_v = torch.flip(logits_v, dims=[2])
    logits_sum += logits_v

    return logits_sum / 3.0


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-patch-root", type=str, default="./test_patches")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-image-path", type=str, default="./data/test/rgb_map3.png")
    parser.add_argument("--output-csv", type=str, default="submission.csv")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--bilinear", action="store_true")
    parser.add_argument("--use-tta", action="store_true")
    parser.add_argument("--save-patch-preds", type=str, default="./test_patch_preds")
    parser.add_argument("--save-full-pred", type=str, default="./test_full_pred.png")
    parser.add_argument("--save-full-color", type=str, default="./test_full_pred_color.png")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_patch_preds, exist_ok=True)

    full_img = Image.open(args.test_image_path).convert("RGB")
    full_w, full_h = full_img.size

    dataset = TestPatchFolderDataset(args.test_patch_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = UNET(n_channels=3, n_classes=8, bilinear=args.bilinear).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    logit_sum = np.zeros((NUM_CLASSES, full_h, full_w), dtype=np.float32)
    weight_sum = np.zeros((full_h, full_w), dtype=np.float32)

    patch_weight = build_weight_map(args.patch_size)

    pbar = tqdm(loader, desc="Inference")
    for images, ys, xs, patch_ids in pbar:
        images = images.to(device, non_blocking=True)
        logits = predict_with_tta(model, images, use_tta=args.use_tta)
        logits = logits.detach().cpu().numpy()   # [B,8,H,W]

        ys = ys.numpy()
        xs = xs.numpy()

        pred_patch = np.argmax(logits, axis=1).astype(np.uint8)

        for i in range(logits.shape[0]):
            y = int(ys[i])
            x = int(xs[i])
            patch_id = patch_ids[i]

            patch_pred_path = os.path.join(args.save_patch_preds, f"{patch_id}.png")
            save_label_map(pred_patch[i], patch_pred_path)

            logit_sum[:, y:y+args.patch_size, x:x+args.patch_size] += logits[i] * patch_weight[None, :, :]
            weight_sum[y:y+args.patch_size, x:x+args.patch_size] += patch_weight

    logit_avg = logit_sum / np.maximum(weight_sum[None, :, :], 1e-6)
    pred_map = np.argmax(logit_avg, axis=0).astype(np.uint8)

    save_label_map(pred_map, args.save_full_pred)
    save_color_map(pred_map, args.save_full_color)

    # only export foreground 1..7
    chw_mask = np.stack(
        [(pred_map == cls_id).astype(np.uint8) for cls_id in range(1, 8)],
        axis=0
    )
    masks_to_submission_rle(args.output_csv, chw_mask)

    print(f"Image size: H={full_h}, W={full_w}")
    print(f"Saved patch predictions to: {args.save_patch_preds}")
    print(f"Saved full prediction map to: {args.save_full_pred}")
    print(f"Saved color prediction map to: {args.save_full_color}")
    print(f"Saved submission to: {args.output_csv}")


if __name__ == "__main__":
    main()