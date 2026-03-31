import os
import random
import argparse
import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import  DataLoader
from tqdm.auto import tqdm

from model.unet.unet import UNET
from data import TrainPatchFolderDataset

IGNORE_INDEX = 255
NUM_CLASSES = 8

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_vis(image_tensor, mask_tensor, pred_tensor, save_path, ignore_index=255):
    img = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)

    gt = mask_tensor.detach().cpu().numpy().astype(np.int32)
    pred = pred_tensor.detach().cpu().numpy().astype(np.int32)

    gt_vis = gt.copy()
    gt_vis[gt_vis == ignore_index] = 0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img)
    axes[0].set_title("Image")
    axes[1].imshow(gt_vis, cmap="tab20", vmin=0, vmax=7)
    axes[1].set_title("GT")
    axes[2].imshow(pred, cmap="tab20", vmin=0, vmax=7)
    axes[2].set_title("Pred")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def compute_class_weights(
    dataset,
    num_classes=8,
    ignore_index=255,
    min_weight=0.2,
    max_weight=5.0,
):
    counts = np.zeros(num_classes, dtype=np.float64)

    print("Computing class weights from training patches...")
    for i in tqdm(range(len(dataset)), desc="Class stats", leave=False):
        _, mask = dataset[i]
        mask = mask.numpy()

        valid = mask != ignore_index
        valid_mask = mask[valid]

        if valid_mask.size == 0:
            continue

        binc = np.bincount(valid_mask, minlength=num_classes)
        counts += binc

    total = counts.sum()
    if total == 0:
        print("Warning: no valid pixels found. Using uniform class weights.")
        return torch.ones(num_classes, dtype=torch.float32)

    freqs = counts / total
    weights = 1.0 / np.sqrt(np.maximum(freqs, 1e-8))
    weights = weights / weights.mean()
    weights = np.clip(weights, min_weight, max_weight)

    print("Class pixel counts:", counts.astype(np.int64).tolist())
    print("Class frequencies:", [round(x, 6) for x in freqs.tolist()])
    print("Class weights:", [round(x, 4) for x in weights.tolist()])

    return torch.tensor(weights, dtype=torch.float32)


def fast_hist(pred, target, num_classes, ignore_index=255):
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]
    hist = np.bincount(
        num_classes * target.astype(int) + pred.astype(int),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return hist


def compute_miou_from_hist(hist):
    intersection = np.diag(hist)
    union = hist.sum(axis=1) + hist.sum(axis=0) - intersection
    iou = intersection / np.maximum(union, 1)
    miou = np.nanmean(iou)
    return miou, iou


def multiclass_dice_loss(logits, target, num_classes=8, ignore_index=255, eps=1e-6):
    """
    logits: [B, C, H, W]
    target: [B, H, W]
    """
    probs = torch.softmax(logits, dim=1)  # [B,C,H,W]

    valid_mask = (target != ignore_index)  # [B,H,W]

    safe_target = target.clone()
    safe_target[~valid_mask] = 0

    one_hot = torch.nn.functional.one_hot(safe_target, num_classes=num_classes)  # [B,H,W,C]
    one_hot = one_hot.permute(0, 3, 1, 2).float()  # [B,C,H,W]

    valid_mask = valid_mask.unsqueeze(1).float()  # [B,1,H,W]
    probs = probs * valid_mask
    one_hot = one_hot * valid_mask

    dims = (0, 2, 3)
    intersection = torch.sum(probs * one_hot, dims)
    denominator = torch.sum(probs, dims) + torch.sum(one_hot, dims)

    dice = (2.0 * intersection + eps) / (denominator + eps)
    dice_loss = 1.0 - dice.mean()

    return dice_loss


def train_one_epoch(model, loader, ce_criterion, optimizer, device, ce_weight, dice_weight):
    model.train()
    total_losses = []
    ce_losses = []
    dice_losses = []

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)

        ce_loss = ce_criterion(logits, masks)
        dice_loss = multiclass_dice_loss(
            logits, masks,
            num_classes=NUM_CLASSES,
            ignore_index=IGNORE_INDEX
        )
        loss = ce_weight * ce_loss + dice_weight * dice_loss

        loss.backward()
        optimizer.step()

        total_losses.append(loss.item())
        ce_losses.append(ce_loss.item())
        dice_losses.append(dice_loss.item())

        pbar.set_postfix(
            total=f"{loss.item():.4f}",
            ce=f"{ce_loss.item():.4f}",
            dice=f"{dice_loss.item():.4f}",
        )

    return (
        float(np.mean(total_losses)),
        float(np.mean(ce_losses)),
        float(np.mean(dice_losses)),
    )


@torch.no_grad()
def validate(model, loader, ce_criterion, device, ce_weight, dice_weight, sample_save_dir=None, epoch=None):
    model.eval()
    total_losses = []
    ce_losses = []
    dice_losses = []
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)

    pbar = tqdm(loader, desc="Val", leave=False)
    for i, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)

        ce_loss = ce_criterion(logits, masks)
        dice_loss = multiclass_dice_loss(
            logits, masks,
            num_classes=NUM_CLASSES,
            ignore_index=IGNORE_INDEX
        )
        loss = ce_weight * ce_loss + dice_weight * dice_loss

        total_losses.append(loss.item())
        ce_losses.append(ce_loss.item())
        dice_losses.append(dice_loss.item())

        pbar.set_postfix(
            total=f"{loss.item():.4f}",
            ce=f"{ce_loss.item():.4f}",
            dice=f"{dice_loss.item():.4f}",
        )

        pred = logits.argmax(dim=1)

        pred_np = pred.detach().cpu().numpy()
        mask_np = masks.detach().cpu().numpy()

        for b in range(pred_np.shape[0]):
            hist += fast_hist(pred_np[b], mask_np[b], NUM_CLASSES, IGNORE_INDEX)

        if sample_save_dir is not None and i == 0:
            save_vis(
                images[0],
                masks[0],
                pred[0],
                os.path.join(sample_save_dir, f"val_epoch_{epoch:03d}.png"),
                ignore_index=IGNORE_INDEX,
            )

    miou, per_class_iou = compute_miou_from_hist(hist)

    return (
        float(np.mean(total_losses)),
        float(np.mean(ce_losses)),
        float(np.mean(dice_losses)),
        float(miou),
        per_class_iou,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-root", type=str, default="./patches")
    parser.add_argument("--save-root", type=str, default="./training_info")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--bilinear", action="store_true")

    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--min-weight", type=float, default=0.2)
    parser.add_argument("--max-weight", type=float, default=5.0)

    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=0.5)

    parser.add_argument("--save-every", type=int, default=1)

    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)

    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(
        args.save_root,
        f"unet8cls_ce_dice_bs{args.batch_size}_{time_str}"
    )
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)

    train_dataset = TrainPatchFolderDataset(
        patch_root=args.patch_root,
        split="train",
        val_ratio=args.val_ratio,
        augment=True,
        seed=args.seed,
    )
    val_dataset = TrainPatchFolderDataset(
        patch_root=args.patch_root,
        split="val",
        val_ratio=args.val_ratio,
        augment=False,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = UNET(n_channels=3, n_classes=8, bilinear=args.bilinear).to(device)

    if args.use_class_weights:
        class_weights = compute_class_weights(
            train_dataset,
            num_classes=NUM_CLASSES,
            ignore_index=IGNORE_INDEX,
            min_weight=args.min_weight,
            max_weight=args.max_weight,
        ).to(device)
        ce_criterion = torch.nn.CrossEntropyLoss(
            ignore_index=IGNORE_INDEX,
            weight=class_weights,
        )
    else:
        ce_criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    best_val = float("inf")
    best_miou = -1.0
    early_stop_counter = 0

    print(f"Train patches: {len(train_dataset)}")
    print(f"Val patches:   {len(val_dataset)}")
    print(f"Save dir: {save_dir}")
    print(f"Loss = {args.ce_weight} * CE + {args.dice_weight} * Dice")

    for epoch in range(1, args.epochs + 1):
        train_total, train_ce, train_dice = train_one_epoch(
            model, train_loader, ce_criterion, optimizer, device,
            args.ce_weight, args.dice_weight
        )

        val_total, val_ce, val_dice, val_miou, per_class_iou = validate(
            model, val_loader, ce_criterion, device,
            args.ce_weight, args.dice_weight,
            sample_save_dir=os.path.join(save_dir, "samples"),
            epoch=epoch,
        )

        scheduler.step(val_total)

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"train_total={train_total:.6f} "
            f"train_ce={train_ce:.6f} "
            f"train_dice={train_dice:.6f} "
            f"val_total={val_total:.6f} "
            f"val_ce={val_ce:.6f} "
            f"val_dice={val_dice:.6f} "
            f"val_mIoU={val_miou:.6f} "
            f"lr={optimizer.param_groups[0]['lr']:.6e}"
        )
        print("Per-class IoU:", [round(float(x), 4) for x in per_class_iou.tolist()])

        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(save_dir, "checkpoints", f"epoch_{epoch:03d}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_total": val_total,
                    "val_ce": val_ce,
                    "val_dice": val_dice,
                    "val_miou": val_miou,
                    "args": vars(args),
                },
                ckpt_path,
            )

        if val_total < best_val:
            best_val = val_total
            best_path = os.path.join(save_dir, "checkpoints", "best_loss.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_total": val_total,
                    "val_ce": val_ce,
                    "val_dice": val_dice,
                    "val_miou": val_miou,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"Saved best-loss model to {best_path}")

        if val_miou > best_miou + args.early_stop_min_delta:
            best_miou = val_miou
            early_stop_counter = 0
            best_path = os.path.join(save_dir, "checkpoints", "best_miou.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_total": val_total,
                    "val_ce": val_ce,
                    "val_dice": val_dice,
                    "val_miou": val_miou,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"Saved best-mIoU model to {best_path}")
        else:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter}/{args.early_stop_patience}")

        if early_stop_counter >= args.early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break


if __name__ == "__main__":
    main()