_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 48  # bs: total bs in all gpus
num_worker = 16
mix_prob = 0.0  # mixes instances in training set
clip_grad = 3.0
empty_cache = False
enable_amp = True
amp_dtype = "bfloat16"

train = dict(type="InsegTrainer")

num_events_train = 1000
num_events_test = 1000
# Weights & Biases specific settings
use_wandb = True  # Enable Weights & Biases logging
wandb_project = "Pretraining-Sonata-PILArNet-M"  # Change to your desired project name
wandb_run_name = f"sonata-pilarnet-inseg-imprint-ft-v1-4GPU-{num_events_train}ev-256patch"  # Descriptive name for this run

# model settings
model = dict(
    type="DefaultInsSegmentor",
    num_classes=1,
    backbone_out_channels=1232,
    backbone=dict(
        type="PT-v3m3",
        in_channels=4,  # [xyz, energy]
        order=("morton", "morton-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 9, 3), 
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(256, 256, 256, 256, 256),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 96, 192, 384),
        dec_num_head=(4, 6, 12, 24),
        dec_patch_size=(256, 256, 256, 256),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=False,
        mask_token=False,
        enc_mode=True,
        freeze_encoder=False,
    ),
    criteria=[
        dict(type="BinaryFocalLoss", loss_weight=1.0, gamma=2.0, alpha=0.8),
    ],
    freeze_backbone=False,
)

# scheduler settings
epoch = 800
eval_epoch = 100
optimizer = dict(type="AdamW", lr=0.0005, weight_decay=0.02)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.0005, 0.000005],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.00005)]

# dataset settings
grid_size = 0.001  # ~ 0.001/(1 / (768.0 * 3**0.5 / 2))
transform = [
    dict(type="NormalizeCoord", center=[384.0, 384.0, 384.0], scale=768.0 * 3**0.5 / 2),
    dict(type="LogTransform", min_val=1.0e-2, max_val=20.0),
    dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", return_grid_coord=True),
    # dict(type="CenterShift", apply_z=True),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
    dict(type="RandomRotate", angle=[-1, 1], axis="x", center=[0, 0, 0], p=0.8),
    dict(type="RandomRotate", angle=[-1, 1], axis="y", center=[0, 0, 0], p=0.8),
    dict(type="RandomFlip", p=0.5),
    # dict(type="RandomJitter", sigma=grid_size / 4, clip=grid_size),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "query_mask", "query_truth"),
        feat_keys=("coord", "energy"),
    ),
]
test_transform = [
    dict(type="NormalizeCoord", center=[384.0, 384.0, 384.0], scale=768.0 * 3**0.5 / 2),
    dict(type="LogTransform", min_val=1.0e-2, max_val=20.0),
    dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", return_grid_coord=True),
    # dict(type="CenterShift", apply_z=True),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "query_mask", "query_truth"),
        feat_keys=("coord", "energy"),
    ),
]


data = dict(
    num_classes=1,
    ignore_index=-1,
    names=['mask'],
    train=dict(
        type="PILArNetH5Dataset",
        split="train",
        data_root="/sdf/home/y/youngsam/data/dune/larnet/h5/DataAccessExamples/",
        transform=transform,
        test_mode=False,
        energy_threshold=0.13,
        min_points=1024,
        max_len=num_events_train,
        remove_low_energy_scatters=False,
        loop=1,
        generate_queries=True,
        num_queries=5,
        query_mask_ratio=0.3,
        ),
    val = dict(
        type="PILArNetH5Dataset",
        split="val",
        data_root="/sdf/home/y/youngsam/data/dune/larnet/h5/DataAccessExamples/",
        transform=test_transform,
        test_mode=False,
        energy_threshold=0.13,
        min_points=1024,
        max_len=num_events_test,
        remove_low_energy_scatters=False,
        generate_queries=True,
        num_queries=5,
        query_mask_ratio=0.3,
        loop=1,
    ),
)


# hook
hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.student.backbone",
        replacement="module.backbone",
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="BinarySegEvaluator", every_n_epochs=1, write_metrics=True),
    dict(type="CheckpointSaver", save_freq=25, evaluator_every_n_epochs=10),
    # dict(type="PreciseEvaluator", test_last=False),
]

