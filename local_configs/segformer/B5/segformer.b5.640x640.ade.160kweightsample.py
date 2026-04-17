_base_ = [
    '../../_base_/models/segformer.py',
    # '../../_base_/datasets/ade20k_repeat.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# data settings
dataset_type = 'MapBackgroundDataset'
data_root = '/cluster/scratch/pangyi/reto/data'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (640, 640)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),# 是否将0类标签减1，变为255作为背景忽略 
    dict(type='Resize', img_scale=(2048, 640), ratio_range=(0.5, 2.0)),
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
        img_scale=(2048, 640),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='AlignedResize', keep_ratio=True, size_divisor=32), # Ensure the long and short sides are divisible by 32
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train_sampler=dict(
        type='WeightedRandomSampler',
        weights_file='/cluster/scratch/pangyi/reto/data/patches_640_split/sample_weights.npy',
        replacement=True),
    train=dict(
        type='RepeatDataset',
        times=50,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='patches_640_split/images/training',
            ann_dir='patches_640_split/annotations/training_background',
            img_suffix='.png',
            seg_map_suffix='.tif', # 
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='patches_640_split/images/validation',
        ann_dir='patches_640_split/annotations/validation_background',
        img_suffix='.png',
        seg_map_suffix='.tif',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='patches_640_split/images/validation',
        ann_dir='patches_640_split/annotations/validation_background',
        # img_dir='test_patches_640_overlap320',
        # ann_dir=None,
        img_suffix='.png',
        seg_map_suffix='.tif',
        pipeline=test_pipeline))

# model settings
# norm_cfg = dict(type='SyncBN', requires_grad=True)
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b5.pth',
    backbone=dict(
        type='mit_b5',
        style='pytorch'),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=8, # 修改类别数
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=768),
        # #交叉熵损失+类别平衡
        # loss_decode=dict( 
        #     type='CrossEntropyLoss',
        #     use_sigmoid=False,
        #     loss_weight=1.0,
        #     class_weight=[0.0089, 0.3343, 0.0192, 1.0651, 4.2411, 1.3391, 0.8709, 0.1213])),

        # 交叉熵损失+Dice损失+类别平衡
        loss_decode=dict(
            type='CEDiceLoss',
            loss_weight=1.0,
            ce_weight=1.0,
            dice_weight=1.0,
            class_weight=[0.1627, 0.9959, 0.2388, 1.7779, 3.5458, 1.9927, 1.6070, 0.6002])), # 修改损失函数权重
        
        # 倒数平方根类别平衡：class_weight=[0.1627, 0.9959, 0.2388, 1.7779, 3.5458, 1.9927, 1.6070, 0.6002]
        # 倒数归一化： class_weight = [0.0089, 0.3343, 0.0192, 1.0651, 4.2411, 1.3391, 0.8709, 0.1213]
    
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

evaluation = dict(interval=4000, metric='mIoU')

