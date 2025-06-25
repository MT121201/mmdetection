_base_ = '../dino/dino-5scale_swin-l_8xb2-12e_coco.py'
load_from = '/home/a3ilab01/treeai/det_tree/weights/12_5_tf12_34_0404.pth'

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
        frozen_stages=4
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
                    dict(type='FocalLossCost', weight=2.0),
                    dict(type='BBoxL1Cost', weight=7.0, box_format='xywh'),
                    dict(type='CIoUCost', weight=3.0)
                ]
            )
        ),
        test_cfg=dict(max_per_img=300)
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
    dict(keep_ratio=True, scale=(640,640), type='Resize'),
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
        # '5_RGB_S_320_pL',
        # '34_RGB_ObjDet_640_pL',
        # '34_RGB_ObjDet_640_pL_b',
        '0_RGB_FullyLabeled/coco'  # assuming already filtered and remapped
    ]
]

train_dataloader = dict(
    _delete_=True,
    batch_size=2,
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='ClassBalancedDataset',
                oversample_thr=1e-3,
                dataset=ds
            ) for ds in datasets
    ]
)
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    num_workers=2,
    persistent_workers=True
)

# train_dataloader = dict(
#     _delete_=True,
#     batch_size=2,
#     dataset=dict(
#         type='ConcatDataset',
#         datasets=datasets  # direct concat of all datasets
#     ),
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     batch_sampler=dict(type='AspectRatioBatchSampler'),
#     num_workers=2,
#     persistent_workers=True
# )


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
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True
)
