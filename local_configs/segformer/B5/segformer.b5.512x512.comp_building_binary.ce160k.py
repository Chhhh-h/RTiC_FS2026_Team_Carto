_base_ = [
    '../../_base_/models/segformer.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

data_root = '/cluster/scratch/caizhi/Segformer/data'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
binary_classes = ('background', 'building')

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(type='AlignedResize', keep_ratio=True, size_divisor=32),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=50,
        dataset=dict(
            type='MapBackgroundDataset',
            data_root=data_root,
            img_dir='patches_512_split/images/training_competition',
            ann_dir='patches_512_split/annotations/training_competition_background',
            img_suffix='.png',
            seg_map_suffix='.tif',
            classes=binary_classes,
            pipeline=train_pipeline)),
    val=dict(
        type='MapBackgroundDataset',
        data_root=data_root,
        img_dir='patches_512_split/images/validation',
        ann_dir='patches_512_split/annotations/validation_background',
        img_suffix='.png',
        seg_map_suffix='.tif',
        classes=binary_classes,
        pipeline=test_pipeline),
    test=dict(
        type='MapBackgroundDataset',
        data_root=data_root,
        img_dir='patches_512_split/images/validation',
        ann_dir='patches_512_split/annotations/validation_background',
        img_suffix='.png',
        seg_map_suffix='.tif',
        classes=binary_classes,
        pipeline=test_pipeline))

norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b5.pth',
    backbone=dict(type='mit_b5', style='pytorch'),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=768),
        loss_decode=dict(
            _delete_=True,
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(custom_keys=dict(
        pos_block=dict(decay_mult=0.0),
        norm=dict(decay_mult=0.0),
        head=dict(lr_mult=10.0))))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mIoU')

work_dir = './work_dirs/segformer_b5_comp_building_binary_ce160k'
