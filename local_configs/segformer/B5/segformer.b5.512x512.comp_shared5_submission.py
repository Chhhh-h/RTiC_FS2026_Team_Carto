_base_ = ['./segformer.b5.512x512.comp_shared5.ce160k.py']

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

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
    test=dict(
        type='MapBackgroundDataset',
        data_root='/cluster/scratch/caizhi/Segformer/data',
        img_dir='patches_512_split/images/test',
        ann_dir=None,
        img_suffix='.png',
        seg_map_suffix='.tif',
        classes=('background', 'river', 'lake', 'wetland', 'stream'),
        pipeline=test_pipeline))
