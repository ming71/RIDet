# model settings
model = dict(
    type='QuadCasRetinaNet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5),
    rbox_head=dict(
        type='QuadCasRetinaHead',
        num_classes=3,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        octave_base_scale=4,
        scales_per_octave=3,
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[8, 16, 32, 64, 128],
        target_means=[.0, .0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0, 1.0],
        loss_ref=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
        loss_odm_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_odm_bbox=dict(type='HungarianLoss', form ='quad', loss_weight=0.01))
        )
# training and testing settings
train_cfg = dict(
    ref_cfg=dict(
        anchor_target_type='hbb_obb_rbox_overlap',
        assigner=dict(
            type='MaxIoUAssignerRbox',
            pos_iou_thr=0.4,
            neg_iou_thr=0.3,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    odm_cfg=dict(
        anchor_target_type='obb_quad_rbox_overlap',
        anchor_inside_type='center',
        calc_offset=False,
        assigner=dict(
            type='MaxIoUAssignerRbox',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False))

test_cfg = dict(
    nms_pre=2000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms_rotated', iou_thr=0.1),  # 15fps
    max_per_img=2000)


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)




train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RotatedResize', img_scale=[(800, 800)], keep_ratio=True),
    dict(type='RotatedRandomFlip', flip_ratio=0.5),
    dict(type='RandomRotate', rate=0.5, angles=[30, 60, 90, 120, 150], auto_bound=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='RotatedResize',img_scale=(800, 800), keep_ratio=True),
            dict(type='RotatedRandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
# dataset settings
dataset_type = 'UCAS_AODDataset'
data_root = 'data/UCAS_AOD/'
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/train.json',
        img_prefix=data_root + 'Train/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/val.json',
        img_prefix=data_root + 'Val/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/test.json',
        img_prefix=data_root + 'Test/images/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='Adam', lr=2*1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[25, 35])
checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 40
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/cascade_retinanet_quad_r50_fpn_hungarian_800_ucas_aod/'
load_from = None
resume_from = None
workflow = [('train', 1)]