_base_ = ["./mask2former_r50_histblock_1024.py"]

work_dir = "work_dirs/histmap_mask2former_r50_ft_best7000_lr1e-5_10k"

# Fine-tune from the best 20k-run checkpoint as weights only.
# Do not use --resume with this config, otherwise optimizer/scheduler state
# from the checkpoint will be restored instead of starting this fine-tune.
load_from = "work_dirs/histmap_mask2former_r50/best_coco_segm_mAP_iter_7000.pth"
resume = False

max_iters = 10000
val_interval = 500

train_cfg = dict(
    type="IterBasedTrainLoop",
    max_iters=max_iters,
    val_interval=val_interval)

optim_wrapper = dict(
    optimizer=dict(lr=1e-5))

param_scheduler = dict(
    type="MultiStepLR",
    begin=0,
    end=max_iters,
    by_epoch=False,
    milestones=[7000, 9000],
    gamma=0.1)

default_hooks = dict(
    logger=dict(type="LoggerHook", interval=50),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        save_last=True,
        save_best="coco/segm_mAP",
        rule="greater",
        max_keep_ckpts=5,
        interval=val_interval))
