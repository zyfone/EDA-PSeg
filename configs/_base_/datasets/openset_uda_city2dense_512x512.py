# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

crop_size = (512, 512)


cityscapes_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='Resize', img_scale=(2048, 1024)),
    dict(type='Resize', img_scale=(1024, 512)),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.5),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.25),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

wildpass2d_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 400)),
    # dict(type='RandomCrop', crop_size=crop_size),
    dict(type='FixScaleRandomCropWH', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 400),
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
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
        type='UDADataset',
        sample_class_stats="/root/autodl-tmp/datasets_seg/datasets/Cityscapes/sample_class_stats_13.json",
        samples_with_class="/root/autodl-tmp/datasets_seg/datasets/Cityscapes/samples_with_class_13.json",
        source=dict(
            type='CityscapesDataset_13',
            data_root="/root/autodl-tmp/datasets_seg/datasets/Cityscapes/",
            img_dir='leftImg8bit/train',
            ann_dir='gtFine/train',
            pipeline=cityscapes_train_pipeline),
        target=dict(
            type='WildPASS2K_13',
            data_root="/root/autodl-tmp/datasets_seg/datasets/WildPASS2K/",
            img_dir='leftImg8bit/',
            ann_dir='gtFine/',
            pipeline=wildpass2d_train_pipeline)),
    val=dict(
        type='DensePASSDataset_13',
        data_root="/root/autodl-tmp/datasets_seg/datasets/DensePASS/",
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline),
    test=dict(
        type='DensePASSDataset_13',
        data_root="/root/autodl-tmp/datasets_seg/datasets/DensePASS/",
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline))
