import torch
from torch import nn
from torch.optim import lr_scheduler
from monai.utils import set_determinism
from torch import optim
from monai.data import CacheDataset, DataLoader, ThreadDataLoader, PersistentDataset
from torch.nn.modules.loss import _Loss
from monai.utils import LossReduction
from monai.losses import DiceLoss, DiceFocalLoss
from monai.transforms import LoadImage
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
import os
from monai.networks.nets import DynUNet
import torch.distributed as dist
from monai.data import partition_dataset

class DiceBceMultilabelLoss(_Loss):
    def __init__(
        self,
        w_dice = 0.5,
        w_bce = 0.5,
        reduction = LossReduction.MEAN,
    ):
        super().__init__(reduction=LossReduction(reduction).value)
        self.w_dice = w_dice
        self.w_bce = w_bce
        self.dice_loss = DiceLoss(sigmoid=True, smooth_nr=0.01, smooth_dr=0.01, include_background=True, batch=True, squared_pred=True)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, label):
        print(pred.shape, label.shape, 'pred and label shape')
        #for dynunet
        if len(pred.size()) - len(label.size()) == 1: 
            pred = torch.unbind(pred, dim=1)
            loss = sum(0.5 ** i * self.dice_loss(p, label) for i, p in enumerate(pred))
            return loss

        loss = self.dice_loss(pred, label) * self.w_dice + self.bce_loss(pred, label) * self.w_bce
        
        return loss

class DiceFocalMultilabelLoss(_Loss):
    def __init__(
        self,
        reduction = LossReduction.MEAN,
    ):
        super().__init__(reduction=LossReduction(reduction).value)
        self.dicefocal_loss = DiceFocalLoss(sigmoid=True, smooth_nr=0.01, smooth_dr=0.01, include_background=True, batch=True, squared_pred=True)
  

    def forward(self, pred, label):
        loss = self.dicefocal_loss(pred, label)
        return loss


class DiceMultilabelLoss(_Loss):
    def __init__(
        self,
        reduction = LossReduction.MEAN,
    ):
        super().__init__(reduction=LossReduction(reduction).value)
        self.dice_loss = DiceLoss(sigmoid=True, smooth_nr=0.01, smooth_dr=0.01, include_background=True, batch=True, squared_pred=True)
  

    def forward(self, pred, label):
        loss = self.dice_loss(pred, label)
        return loss

def get_train_dataloader(train_dataset, cfg):

    if cfg.gpu_cache:
        train_dataloader = ThreadDataLoader(
            train_dataset,
            shuffle=True,
            batch_size=cfg.batch_size,
            num_workers=0,
            drop_last=True,
        )
        return train_dataloader 

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=True,
    )
    return train_dataloader


def get_val_dataloader(val_dataset, cfg):
    if cfg.val_gpu_cache:
        val_dataloader = ThreadDataLoader(
            val_dataset,
            batch_size=cfg.val_batch_size,
            num_workers=0,
        )
        return val_dataloader

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.val_batch_size,
        num_workers=cfg.num_workers,
    )
    return val_dataloader



def get_train_dataset(cfg):

    #datalist = cfg.data_json['train']

    ##using all data to train
    datalist = cfg.data_json['train'] + cfg.data_json['val']
 
    if cfg.multigpu:
        datalist = partition_dataset(
            data=datalist,
            shuffle=True,
            num_partitions=dist.get_world_size(),
            even_divisible=True,
        )[dist.get_rank()]
        train_ds = CacheDataset(
        data=datalist,
        transform=cfg.train_transforms,
        cache_rate=cfg.train_cache_rate,
        num_workers=cfg.num_workers,
        copy_cache=False,
        )
    else:
        train_ds = CacheDataset(
            data=datalist,
            transform=cfg.train_transforms,
            cache_rate=cfg.train_cache_rate,
            num_workers=cfg.num_workers,
            copy_cache=False,
        )

    return train_ds



def get_val_dataset(cfg):

    datalist = cfg.data_json['val']

    if cfg.multigpu:
        datalist = partition_dataset(
            data=datalist,
            shuffle=False,
            num_partitions=dist.get_world_size(),
            even_divisible=False,
        )[dist.get_rank()]
    
        val_ds = CacheDataset(
            data=datalist,
            transform=cfg.val_transforms,
            cache_rate=cfg.val_cache_rate,
            num_workers=cfg.num_workers,
            copy_cache=False,
        )

    else:
        val_ds = CacheDataset(
            data=datalist,
            transform=cfg.val_transforms,
            cache_rate=cfg.val_cache_rate,
            num_workers=cfg.num_workers,
            copy_cache=False,
        )

    return val_ds


def get_optimizer(model, cfg):

    params = model.parameters()
    #optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    optimizer = optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    return optimizer

def get_scheduler(cfg, optimizer, total_steps, epoch):

    if cfg.lr_mode == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.epochs * (total_steps // cfg.batch_size),
            eta_min=cfg.min_lr,
        )

    elif cfg.lr_mode == "warmup_restart":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.restart_epoch * (total_steps // cfg.batch_size),
            T_mult=1,
            eta_min=cfg.min_lr,
        )

    elif cfg.lr_mode == "lambda":
        scheduler = lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (1 - epoch / (total_steps * cfg.epochs)) ** 0.9
        )

    return scheduler


def create_checkpoint(model, optimizer, epoch, scheduler=None, scaler=None):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    return checkpoint

def get_kernels_strides(cfg):
    sizes, spacings = cfg.roi_size, cfg.spacing    
    input_size=sizes
    strides, kernels = [], []

    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [
            2 if ratio <= 2 and size >= 8 else 1
            for (ratio, size) in zip(spacing_ratio, sizes)
        ]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(sizes, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {input_size[idx]} in the spatial dimension {idx}."
                )
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
   
    return kernels, strides

