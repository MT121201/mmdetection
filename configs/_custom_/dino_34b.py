_base_ = '../dino/dino-5scale_swin-l_8xb2-12e_coco.py'

max_epochs = 36
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1
)
load_from = '/home/a3ilab01/treeai/det_tree/34a_best.pth'


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
data_root = '/home/a3ilab01/treeai/dataset/34_RGB_ObjDet_640_pL_b/'  

classes = (
    'cls_3', 'cls_5', 'cls_6', 'cls_9', 'cls_11', 'cls_12', 'cls_13',
    'cls_24', 'cls_25', 'cls_26', 'cls_30', 'cls_36', 'cls_37',
    'cls_43', 'cls_48', 'cls_50', 'cls_53', 'cls_56', 'cls_58',
    'cls_59', 'cls_60'
)

metainfo=dict(classes=classes)
model = dict(
    bbox_head=dict(
        num_classes=len(classes)
    )
)

train_pipeline = [
    dict(
        type='MixUp',
        img_scale=(800, 800),
        ratio_range=(0.8, 1.2),
        pad_val=114.0
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]

train_dataloader = dict(
    _delete_=True,
    batch_size=2,
    dataset=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type='MultiImageMixDataset',
            dataset=dict(
                type='ClassBalancedDataset',
                oversample_thr=1e-3,
                dataset=dict(
                    type='CocoDataset',
                    data_root='/home/a3ilab01/treeai/dataset/34_RGB_ObjDet_640_pL_b',
                    ann_file='annotations/train.json',
                    data_prefix=dict(img='images/train/'),
                    metainfo=dict(classes=classes),
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True)
                    ]
                )
            ),
            pipeline=train_pipeline
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
        data_root='/home/a3ilab01/treeai/dataset/34_RGB_ObjDet_640_pL_b',
        ann_file='/home/a3ilab01/treeai/dataset/34_RGB_ObjDet_640_pL_b/annotations/val.json',
        data_prefix=dict(img='images/val/'),
        metainfo=dict(classes=classes),
        pipeline=test_pipeline,
        test_mode=True
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
    num_workers=2,
    persistent_workers=True
)
test_dataloader = val_dataloader

test_evaluator = dict(
    ann_file=data_root + 'annotations/val.json',
    metric='bbox',
    type='CocoMetric'
)

val_evaluator = dict(
    ann_file=data_root + 'annotations/val.json',
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
        save_best='coco/bbox_mAP_50',  # <- USE mAP@50
        rule='greater'
    ),
    logger=dict(type='LoggerHook', interval=50)
)

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP_50',  # <- USE mAP@50
        rule='greater',
        patience=5,
        min_delta=0.001,
    )
]