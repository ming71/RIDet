# fp16 settings
# fp16 = dict(loss_scale=512.)

# model settings
model = dict(
    type='S2ANetDetector',
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
        type='S2ANetHead',
        num_classes=6,
        in_channels=256,
        feat_channels=256,
        stacked_convs=2,
        align_conv_type='AlignConv',#[AlignConv,DCN,GA_DCN]
        align_conv_size=3,
        anchor_ratios=[1.0],
        anchor_strides=[8, 16, 32, 64, 128],
        anchor_scales=[4],
        target_means=[.0, .0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0, 1.0],
        loss_fam_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_fam_bbox=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_odm_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_odm_bbox=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)))
# training and testing settings
train_cfg = dict(
    fam_cfg=dict(
        anchor_target_type='hbb_obb_rbox_overlap',
        assigner=dict(
            type='MaxIoUAssignerRbox',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    odm_cfg=dict(
        anchor_target_type='obb_obb_rbox_overlap',
        anchor_inside_type='center',
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
#     score_thr=0.05,  # confidence 0.3
    score_thr=0.3,  # confidence 0.3
    nms=dict(type='nms_rotated', iou_thr=0.1),  
    max_per_img=2000)


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RotatedResize', img_scale=(1024, 1024), keep_ratio=True),
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
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RotatedResize', img_scale=(1024, 1024), keep_ratio=True),
            dict(type='RotatedRandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# dataset settings
dataset_type = 'RAChallengeDataset'
data_root = 'data/RAChallenge/stage1/train/'  # train无augmentation
aug_data_root = 'data/RAChallenge/stage1/train_augment/'  # train with augmentation
train_merge_data_root = 'data/RAChallenge/stage1/train_merge/'
warmup_merge_data_root = 'data/RAChallenge/warmup_merge/'  # warmup_aug + warmup
warmup_data_root = 'data/RAChallenge/warmup/'  # warmup数据无augmentation
test_root = 'data/RAChallenge/stage1/'

data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=2,
    
    # train + train_aug
    train=dict(
        type=dataset_type,
        ann_file=train_merge_data_root + 'train.json',
        img_prefix=train_merge_data_root + 'images/',
        pipeline=train_pipeline),
    
#     # warmup_aug + warmup
#     train=dict(
#         type=dataset_type,
#         ann_file=warmup_merge_data_root + 'train.json',
#         img_prefix=warmup_merge_data_root + 'images/',
#         pipeline=train_pipeline),
    
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval_split/trainval.json',
        img_prefix=data_root + 'trainval_split/images/',
        pipeline=test_pipeline),
    
    # submission
#     test=dict(
#         type=dataset_type,
#         ann_file=test_root + 'test.json',
#         img_prefix=test_root + 'test1/',
#         pipeline=test_pipeline)
    
    ## eval
    test=dict(
        type=dataset_type,
        ann_file=warmup_data_root + 'train.json',
        img_prefix=warmup_data_root + 'images/',
        pipeline=test_pipeline)
    
        )
# optimizer
optimizer = dict(type='Adam', lr=1e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
#     step=[15, 30])
#     step=[20, 35, 50])
    step=[10, 20, 30])
checkpoint_config = dict(interval=12)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 36
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/s2anet_r50_fpn_1024_aug_RA/'
load_from = 'work_dirs/s2anet_r50_fpn_1024_aug_RA/epoch_60.pth'
resume_from = None
workflow = [('train', 1)]


