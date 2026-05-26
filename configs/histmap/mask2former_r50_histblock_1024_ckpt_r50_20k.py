_base_ = ["./mask2former_r50_histblock_1024.py"]

work_dir = "work_dirs/histmap_mask2former_ckpt_r50_20k"
load_from = "pretrained/mask2former_r50_8xb2-lsj-50e_coco.pth"
resume = False

max_iters = 20000
val_interval = 1000
train_cfg = dict(
    type="IterBasedTrainLoop",
    max_iters=max_iters,
    val_interval=val_interval)

param_scheduler = dict(
    type="MultiStepLR",
    begin=0,
    end=max_iters,
    by_epoch=False,
    milestones=[16000, 19000],
    gamma=0.1)

default_hooks = dict(
    checkpoint=dict(interval=val_interval))
