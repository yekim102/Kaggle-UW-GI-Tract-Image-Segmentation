import numpy as np
import torch
import os
from types import SimpleNamespace
from monai.transforms import (
    Compose,
    LoadImaged,
    RandSpatialCropd,
    EnsureTyped,
    CastToTyped,
    NormalizeIntensityd,
    RandFlipd,
    CenterSpatialCropd,
    ScaleIntensityRanged,
    RandAffined,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandCoarseDropoutd,
    Rand2DElasticd,
    Lambdad,
    Resized,
    AddChanneld,
    RandGaussianNoised,
    RepeatChanneld,
    Transposed,
    OneOf,
    EnsureChannelFirstd,
    RandLambdad,
    Spacingd,
    FgBgToIndicesd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    ToDeviced,
    SpatialPadd,
    RandGridDistortiond,
    RandZoomd,
    RandGaussianSmoothd,
    RandHistogramShiftd,
    RandAdjustContrastd,
    Rand3DElasticd,
    RandRotated,
    RandBiasFieldd,
    RandGibbsNoised,
    RandCoarseShuffled,
)


cfg = SimpleNamespace(**{})

# data path
cfg.data_dir = "/home/synergy/yhk/GI/"
cfg.fold = 0

cfg.test_df = cfg.data_dir + "sample_submission.csv"

# training
cfg.seed = -1
cfg.start_cal_metric_epoch = 1
cfg.warmup = 1

# ressources
cfg.mixed_precision = True
cfg.device = "cuda"
cfg.num_workers = 12
cfg.weights = None

# train
cfg.train = True
cfg.eval = True
cfg.eval_epochs = 5
cfg.start_eval_epoch = 0  # when use large lr, can set a large num
cfg.current_epoch = 1108
cfg.run_tta_val = False
cfg.load_best_weights = False
cfg.amp = False
cfg.val_amp = False
# lr
# warmup_restart, cosine
cfg.lr_mode = "lambda"
cfg.lr = 0.007
cfg.min_lr = 1e-6
cfg.weight_decay = 1e-6
cfg.epochs = 1500
cfg.restart_epoch = 100  # only for warmup_restart
cfg.current_training_loss = 0
cfg.finetune_lb = -1

# dataset
cfg.img_size = (384, 384, 144)
cfg.spacing = (1.5, 1.5, 1)
cfg.batch_size = 2
cfg.val_batch_size = 1
cfg.train_cache_rate = 0
cfg.val_cache_rate = 0
cfg.gpu_cache = False
cfg.val_gpu_cache = False

# val
cfg.roi_size = (224, 224, 80)
cfg.sw_batch_size = 4


cfg.train_transforms = Compose(
    [
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        RandCropByPosNegLabeld(
            keys=["image", "mask"],
            label_key="mask",
            spatial_size=cfg.roi_size,
            pos=3,
            neg=1,
            num_samples=3,
            image_key="image",
            image_threshold=0,
        ),
        Lambdad(keys="image", func=lambda x: x / x.max()),
        RandZoomd(
            keys=["image", "mask"],
            min_zoom=0.9,
            max_zoom=1.2,
            mode=("trilinear", "nearest"),
            align_corners=(True, None),
            prob=0.15,
        ),
        RandGaussianNoised(keys=["image"], std=0.01, prob=0.15),
        RandGaussianSmoothd(
            keys=["image"],
            sigma_x=(0.5, 1.15),
            sigma_y=(0.5, 1.15),
            sigma_z=(0.5, 1.15),
            prob=0.15,
        ),
        RandFlipd(keys=("image", "mask"), prob=0.5, spatial_axis=[0]),
        RandFlipd(keys=("image", "mask"), prob=0.5, spatial_axis=[1]),
        RandAffined(
            keys=("image", "mask"),
            prob=0.5,
            rotate_range=np.pi / 12,
            translate_range=(cfg.roi_size[0]*0.0625, cfg.roi_size[1]*0.0625),
            scale_range=(0.1, 0.1),
            mode="nearest",
            padding_mode="reflection",
        ),
        OneOf(
            [
                RandGridDistortiond(keys=("image", "mask"), prob=0.5, distort_limit=(-0.05, 0.05), mode="nearest", padding_mode="reflection"),
                RandCoarseDropoutd(
                    keys=("image", "mask"),
                    holes=5,
                    max_holes=8,
                    spatial_size=(1, 1, 1),
                    max_spatial_size=(12, 12, 12),
                    fill_value=0.0,
                    prob=0.5,
                ),
            ]
        ),
        RandHistogramShiftd(keys=['image'], num_control_points=3, prob=0.2),
        RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.5, 1)),
        Rand3DElasticd(keys=['image', 'mask'], sigma_range=(5,7), mode=['bilinear', 'nearest'], prob=0.5, magnitude_range=(50, 150), padding_mode='zeros'),
        RandScaleIntensityd(keys="image", factors=(-0.2, 0.2), prob=0.5),
        RandShiftIntensityd(keys="image", offsets=(-0.1, 0.1), prob=0.5),
        EnsureTyped(keys=("image", "mask"), dtype=torch.float32),
    ]
)


cfg.val_transforms = Compose(
    [
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        #SpatialPadd(keys=["image", "mask"], spatial_size=cfg.img_size),
        Lambdad(keys="image", func=lambda x: x / x.max()),
        EnsureTyped(keys=("image", "mask"), dtype=torch.float32),
    ]
)


