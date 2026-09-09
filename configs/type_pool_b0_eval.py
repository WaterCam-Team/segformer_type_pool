# SegFormer-B0 + DenseCLIP water-type pool -- EVALUATION ONLY.
#
# Self-contained: no _base_ chain, no optimizer / schedule / train pipeline.
# On a new machine only the four paths in the DATA ROOTS block need changing.

# ---------------------------------------------------------------- DATA ROOTS
data_root_normal_flood      = '/data1/huantao/workspace/project/flood_seg/dataset_w_urban_flood/my_normal_flood_huantao_w_urban_flood/'
data_root_challenging_flood = '/data1/huantao/workspace/project/flood_seg/dataset_w_urban_flood/my_challenging_flood_huantao_w_urban_flood_new/'
data_root_challenging_water = '/data1/huantao/workspace/project/flood_seg/dataset_setting2/my_waterdataset_val_edge_setting2/'

# --------------------------------------------------------------------- MODEL
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='EncoderDecoder_denseclip_attribute',
    pretrained=None,                       # weights come from --checkpoint
    backbone=dict(type='mit_b0', style='pytorch'),
    decode_head=dict(
        type='SegFormerHead_denseclip',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        use_text=False,
        # required by SegFormerHead_denseclip.__init__, unused by this variant
        K_water=3,
        K_bg=3,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    classnames=['background', 'water'],
    # required by EncoderDecoder_denseclip_attribute.__init__, unused by this
    # variant (the type pool is sized by CHALLENGING_TYPES, and the prompt
    # learner uses its own context_len=8 default)
    K_water=3,
    K_bg=3,
    context_length=8,
    test_cfg=dict(mode='whole'))

# ---------------------------------------------------------------------- DATA
dataset_type = 'WaterDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# Order in which the subsets are evaluated and reported.
eval_sets = ['test_normal_flood', 'test_challenging_flood', 'test_challenging_water']

data = dict(
    workers_per_gpu=2,
    test_normal_flood=dict(
        type=dataset_type,
        data_root=data_root_normal_flood,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        pipeline=test_pipeline),
    test_challenging_flood=dict(
        type=dataset_type,
        data_root=data_root_challenging_flood,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        pipeline=test_pipeline),
    test_challenging_water=dict(
        type=dataset_type,
        data_root=data_root_challenging_water,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        pipeline=test_pipeline))

cudnn_benchmark = True
