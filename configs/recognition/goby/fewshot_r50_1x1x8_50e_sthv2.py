# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet',
        pretrained='torchvision://resnet50',
        depth=50,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True),
    cls_head=dict(
        # for backbone training
        # type='TSNHead',
        # num_classes=64,

        # for fewshot training
        type='FewShotHead',
        num_classes=5,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.01))
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips=None)
# dataset settings
dataset_type = 'VideoDataset'
data_root = '/data1/wrye/dataset/sthv2/raw-part/20bn-something-something-v2'
data_root_val = '/data1/wrye/dataset/sthv2/raw-part/20bn-something-something-v2'
ann_file_train = 'data/sthv2100/train_mmaction_convert.list'
ann_file_val = 'data/sthv2100/test_mmaction_convert.list'
ann_file_test = 'data/sthv2100/test_mmaction_convert.list'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='TenCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    # for backbone training
    # videos_per_gpu=6,
    # workers_per_gpu=4,

    # for fewshot training
    videos_per_gpu=1,
    workers_per_gpu=4,

    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
# optimizer 
optimizer_backbone_training = dict(
    type='SGD', lr=[0.0001,0.001], momentum=0.9,
    weight_decay=0.0005)  

optimizer_fewshot_training = dict(
    type='SGD', lr=0.001, momentum=0.9,
    # type='SGD', lr=0.00002, momentum=0.9,
    # type='SGD', lr=0, momentum=0.9,
    weight_decay=0.0005)  

optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[20, 30, 40])
total_epochs = 6
checkpoint_config = dict(interval=1)
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/resnet_retrain_GT_6epoch/'

# for backbone training
# load_from = None

# for fewshot training
load_from = './work_dirs/resnet/epoch_6.pth'

resume_from = None
workflow = [('train', 1)]

# few-shot setting
n_way =  5   
k_shot = 1
train_episode = 2000
val_episode = 2400 
training_dataset_class_num = 64
test_dataset_class_num = 24

fewshot_training = True
# fewshot_training = False