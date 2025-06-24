_base_ = '/home/a3ilab01/treeai/mmdetection/work_dirs/dino_34b/best_coco_bbox_mAP_50_epoch_20.pth'

max_epochs = 36
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1
)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1
    )
]

dataset_type = 'CocoDataset'
data_root = '/home/a3ilab01/treeai/dataset/'

classes = tuple(f'cls_{i}' for i in range(1, 62))
metainfo = dict(classes=classes)

model = dict(
    bbox_head=dict(
        num_classes=len(classes)
    )
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='PackDetInputs')
]


test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(1333, 800), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
        ),
        type='PackDetInputs'
    ),
]

datasets = [
    dict(
        type='CocoDataset',
        data_root=f'{data_root}{subdir}/',
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/train/'),
        metainfo=metainfo,
        pipeline=train_pipeline
    )
    for subdir in [
        '12_RGB_ObjDet_640_fL',
        '5_RGB_S_320_pL',
        '34_RGB_ObjDet_640_pLa',
        '34_RGB_ObjDet_640_pLb',
        '0_RGB_fL/coco'  # assuming already filtered and remapped
    ]
]

train_dataloader = dict(
    _delete_=True,
    batch_size=2,
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type='ClassBalancedDataset',
            oversample_thr=1e-2,
            dataset=dict(
                type='ConcatDataset',
                datasets=datasets
            )
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
