"""
PointNext-S configuration for PILArNet dataset

This config file is designed to train a PointNext-S model on the PILArNet dataset
for semantic segmentation of particle physics point clouds.
"""

_base_ = ["../_base_/default_runtime.py"]

# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="PointNext",
        in_channels=4,  # [xyz, energy]
        encoder_depths=[2, 2, 2, 6, 2],
        encoder_channels=[32, 64, 128, 256, 512],
        decoder_depths=[2, 2, 2, 2],
        decoder_channels=[128, 128, 128, 128],
        conv_cfg=dict(type="Conv1d"),
        norm_cfg=dict(type="BN1d", momentum=0.02),
        act_cfg=dict(type="GELU"),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=0.2, ignore_index=-1),
    ],
)

# scheduler settings
epoch = 100
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# dataset settings
data = dict(
    num_classes=5,  # Number of semantic classes in PILArNet
    ignore_index=-1,
    names=[
        "shower", 
        "track", 
        "michel", 
        "delta", 
        "low_energy"
    ],
    train=dict(
        type="PILArNetDataset",
        split="train",
        data_root="data/pilarnet",
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomScale", scale=[0.8, 1.2]),
            dict(
                type="RandomRotation",
                angle=[-1, 1],
                axis="z",
                center=[0, 0, 0],
                p=0.5,
            ),
            dict(type="RandomSymmetries", symmetries=[0, 0, 0], p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="ElasticDistortion",
                distortion_params=[[0.2, 0.4], [0.8, 1.6]],
                p=0.5,
            ),
            dict(type="RandomMixup", k=3, p=0.5),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_min_coord=True,
                return_displacement=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment"),
                feat_keys=("color", "energy", "displacement"),
            ),
        ],
        loop=1,
    ),
    val=dict(
        type="PILArNetDataset",
        split="val",
        data_root="data/pilarnet",
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_min_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment"),
                feat_keys=("color", "energy"),
            ),
        ],
        loop=1,
    ),
    test=dict(
        type="PILArNetDataset",
        split="val",
        data_root="data/pilarnet",
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment"),
                feat_keys=("color", "energy"),
            ),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color", "energy"),
                return_discrete_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=True),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "index"),
                    feat_keys=("color", "energy"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9, 0.9])],
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1, 1])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomScale", scale=[1.1, 1.1])],
            ],
        ),
    ),
)

# training settings
batch_size = 8
num_worker = 4
train_gpu_batch_size = 8
train = dict(
    type="MultiDatasetDataloader",
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            data["train"],
        ],
        loop=1,
    ),
    batch_size=train_gpu_batch_size,
    num_workers=num_worker,
    mix_prob=0,
)

# validation settings
val = dict(
    type="MultiDatasetDataloader",
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            data["val"],
        ],
        loop=1,
    ),
    batch_size=train_gpu_batch_size,
    num_workers=num_worker,
)

# test settings
test = dict(
    type="MultiDatasetDataloader",
    dataset=data["test"],
    batch_size=train_gpu_batch_size,
    num_workers=num_worker,
)

# logging settings
log_level = "INFO"
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHook"),
    ],
)

# evaluation settings
eval_config = dict(interval=1, save_best="miou")
eval_pipelines = [
    dict(
        type="DefaultEvaluator",
        metric_names=["miou", "acc"],
    ),
]

# reproducibility
deterministic = True
cudnn_benchmark = False 