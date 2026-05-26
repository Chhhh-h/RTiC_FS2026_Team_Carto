_base_ = ["../mask2former/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco.py"]

metainfo = dict(classes=("building_block",), palette=[(220, 60, 60)])
data_root = "data/histmap_instance/"
work_dir = "work_dirs/histmap_mask2former_ckpt_swin_t_20k"
load_from = "pretrained/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco.pth"
resume = False

num_things_classes = 1
num_stuff_classes = 0
num_classes = 1
image_size = (1024, 1024)

model = dict(
    backbone=dict(init_cfg=None),
    panoptic_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=300,
        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1])),
    panoptic_fusion_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes),
    test_cfg=dict(
        panoptic_on=False,
        semantic_on=False,
        instance_on=True,
        max_per_image=300,
        iou_thr=0.8,
        filter_low_score=True))

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="annotations/instances_train.json",
        data_prefix=dict(img="images/train/"),
        filter_cfg=dict(filter_empty_gt=True, min_size=32)))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="annotations/instances_val.json",
        data_prefix=dict(img="images/val/")))

test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type="CocoMetric",
    ann_file=data_root + "annotations/instances_val.json",
    metric=["bbox", "segm"],
    format_only=False)

test_evaluator = val_evaluator

max_iters = 20000
val_interval = 1000
train_cfg = dict(
    type="IterBasedTrainLoop",
    max_iters=max_iters,
    val_interval=val_interval)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

param_scheduler = dict(
    type="MultiStepLR",
    begin=0,
    end=max_iters,
    by_epoch=False,
    milestones=[16000, 19000],
    gamma=0.1)

default_hooks = dict(
    logger=dict(type="LoggerHook", interval=50),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        save_last=True,
        save_best="coco/segm_mAP",
        rule="greater",
        max_keep_ckpts=3,
        interval=val_interval))

auto_scale_lr = dict(enable=False, base_batch_size=16)
