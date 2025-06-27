_base_ = '../dino/dino-5scale_swin-l_8xb2-12e_coco.py'
load_from = '/home/a3ilab01/treeai/det_tree/weights/12_0_049.pth'

max_epochs = 36
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1
)

# LR Scheduler with warmup
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=True, begin=0, end=1),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs,
        eta_min=1e-6,
        by_epoch=True
    )
]

# Optimizer: AdamW with reduced backbone LR
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05),
    accumulative_counts=4,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.5)}
    )
)

dataset_type = 'CocoDataset'
data_root = '/home/a3ilab01/treeai/dataset/'

dataset12_classes = [
    "betula papyrifera", "tsuga canadensis", "picea abies", "acer saccharum",
    "betula sp.", "pinus sylvestris", "picea rubens", "betula alleghaniensis",
    "larix decidua", "fagus grandifolia", "picea sp.", "fagus sylvatica",
    "dead tree", "acer pensylvanicum", "populus balsamifera", "quercus ilex",
    "quercus robur", "pinus strobus", "larix laricina", "larix gmelinii",
    "pinus pinea", "populus grandidentata", "pinus montezumae", "abies alba",
    "betula pendula", "pseudotsuga menziesii", "fraxinus nigra",
    "dacrydium cupressinum", "cedrus libani", "acer pseudoplatanus",
    "pinus elliottii", "cryptomeria japonica", "pinus koraiensis",
    "abies holophylla", "alnus glutinosa", "fraxinus excelsior", "coniferous",
    "eucalyptus globulus", "pinus nigra", "quercus rubra", "tilia europaea",
    "abies firma", "acer sp.", "metrosideros umbellata", "acer rubrum",
    "picea mariana", "abies balsamea", "castanea sativa", "tilia cordata",
    "populus sp.", "crataegus monogyna", "quercus petraea", "acer platanoides"
]

metainfo = dict(classes=dataset12_classes)

model = dict(
    backbone=dict(
        frozen_stages=-1,
        with_cp=False
    ),
    bbox_head=dict(
        type='DINOHead',
        num_classes=len(dataset12_classes),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=7.0),
        loss_iou=dict(type='CIoULoss', loss_weight=3.0),
        train_cfg=dict(
            assigner=dict(
                type='HungarianAssigner',
                match_costs=[
                    dict(type='FocalLossCost', weight=3.0),
                    dict(type='BBoxL1Cost', weight=3.0, box_format='xywh'),
                    dict(type='CIoUCost', weight=3.0)
                ]
            )
        ),
        test_cfg=dict(max_per_img=300)
    )
)


test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(640,640), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
        ),
        type='PackDetInputs'
    ),
]
# === Pre-transform just for loading required keys like 'gt_bboxes' ===
pre_transform = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
]

# === Full train pipeline applied by MultiImageMixDataset ===
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MixUp',
        img_scale=(640, 640),
        ratio_range=(0.8, 1.6),
        flip_ratio=0.5,
        pad_val=114,
        max_iters=15,
        bbox_clip_border=True
    ),
    dict(type='AutoAugment', policies=[
        [dict(type='Resize', scale=(640, 640), keep_ratio=True)],
        [
            dict(type='Resize', scale=(512, 512), keep_ratio=True),
            dict(type='RandomCrop', crop_size=(512, 512), allow_negative_crop=True),
            dict(type='Resize', scale=(640, 640), keep_ratio=True)
        ]
    ]),
    dict(type='PhotoMetricDistortion'),
    dict(type='CutOut', n_holes=2, cutout_shape=(64, 64)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

# === Dataset definition with minimal pipeline ===
datasets = [
    dict(
        type='CocoDataset',
        data_root=f'{data_root}{subdir}/',
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/train/'),
        metainfo=metainfo,
        pipeline=pre_transform  # ✅ critical fix here
    )
    for subdir in [
        '12_RGB_ObjDet_640_fL',
        '0_RGB_FullyLabeled/coco'  
    ]
]

# === Final train_dataloader ===
train_dataloader = dict(
    _delete_=True,
    batch_size=2,
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type='MultiImageMixDataset',  # ✅ required for MixUp
            dataset=dict(
                type='ConcatDataset',
                datasets=[
                    dict(
                        type='ClassBalancedDataset',
                        oversample_thr=1e-3,
                        dataset=ds
                    ) for ds in datasets
                ]
            ),
            pipeline=train_pipeline  # ✅ apply full MixUp + aug pipeline here
        )
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    num_workers=2,
    persistent_workers=True
)




val_dataloader = dict(
    _delete_=True,
    batch_size=2,
    dataset=dict(
        type='CocoDataset',
        data_root=f'{data_root}12_RGB_ObjDet_640_fL/',
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/val/'),
        metainfo=metainfo,
        pipeline=test_pipeline,
        test_mode=True
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
    num_workers=2,
    persistent_workers=True
)
test_dataloader = val_dataloader

test_evaluator = dict(
    ann_file=f'{data_root}12_RGB_ObjDet_640_fL/annotations/val.json',
    metric='bbox',
    type='CocoMetric'
)

val_evaluator = dict(
    ann_file=f'{data_root}12_RGB_ObjDet_640_fL/annotations/val.json',
    metric='bbox',
    type='CocoMetric'
)

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer'
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='coco/bbox_mAP_50',
        rule='greater'
    ),
    logger=dict(type='LoggerHook', interval=50)
)

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP_50',
        rule='greater',
        patience=5,
        min_delta=0.001,
    )
]
custom_hooks.append(
    dict(
        type='EMAHook',
        momentum=0.0002,
        priority='ABOVE_NORMAL'
    )
)

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=False  # ✅ turn off unused param tracking
)

auto_scale_lr = dict(enable=True, base_batch_size=16)
