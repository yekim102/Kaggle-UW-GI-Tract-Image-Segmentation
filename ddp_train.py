#modified ddp from https://github.com/Project-MONAI/tutorials/blob/main/acceleration/distributed_training/brats_training_ddp.py
#modified training and inference code from https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/discussion/325646
import argparse
import gc
import importlib
import os
import sys
import shutil

import numpy as np
import pandas as pd
import torch
from torch import nn
from monai.handlers.utils import from_engine
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from monai.losses import DiceFocalLoss
from utils import *

from monai.transforms import (
    Compose,
    Activations,
    AsDiscrete,
    Activationsd,
    AsDiscreted,
    KeepLargestConnectedComponentd,
    Invertd,
    LoadImage,
    Transposed,
)
import json
from metric import HausdorffScore
from monai.utils import set_determinism
from monai.losses import DiceLoss, DiceCELoss
from monai.networks.nets import UNet, SwinUNETR, AttentionUnet
from monai.optimizers import Novograd
from monai.metrics import DiceMetric
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from monai.metrics import compute_meandice
import wandb

def main_worker(cfg):

    # # set random seed
    set_determinism(cfg.seed)

    if cfg.local_rank != 0:
        f= open(os.devnull, 'w')
        sys.stdout = sys.stderr = f

    if cfg.multigpu:
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{cfg.local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device('cuda')

    # data sequence
    cfg.data_json_dir = cfg.data_dir + f"dataset_3d_fold_{cfg.fold}.json"

  
    with open(cfg.data_json_dir, "r") as f:
        cfg.data_json = json.load(f)

    train_dataset = get_train_dataset(cfg)
    val_dataset = get_val_dataset(cfg)

    train_dataloader = get_train_dataloader(train_dataset, cfg)
    val_dataloader = get_val_dataloader(val_dataset, cfg)

    
    print(f"run fold {cfg.fold}, train len: {len(train_dataset)}")

    # n_class = 3
    # in_channels = 1
    # kernels, strides = get_kernels_strides(cfg)

    # model = DynUNet(
    #     spatial_dims=3,
    #     in_channels=in_channels,
    #     out_channels=n_class,
    #     kernel_size=kernels,
    #     strides=strides,
    #     upsample_kernel_size=strides[1:],
    #     norm_name="BATCH",
    #     deep_supervision=True,
    #     deep_supr_num=3,
    #     res_block=True
    # ).to(device)

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        kernel_size=3,
        up_kernel_size=3,
        num_res_units=4,
        act="PRELU",
        norm="instance",
        dropout=0.2,
        bias=True,
        dimensions=None,
    ).to(device)

       

    # model = SwinUNETR(
    #     img_size=cfg.roi_size, 
    #     in_channels=1, 
    #     out_channels=3, 
    #     depths=(2, 2, 2, 2), 
    #     num_heads=(3, 6, 12, 24), 
    #     feature_size=24, 
    #     norm_name='instance',    #


    #     drop_rate=0.2, 
    #     attn_drop_rate=0.2, 
    #     dropout_path_rate=0.0, 
    #     normalize=True, 
    #     use_checkpoint=False, 
    #     spatial_dims=3).to(device)



   #model = torch.nn.DataParallel(model)
    
    print(cfg.weights)
    if cfg.weights is not None:
        model.load_state_dict(torch.load(cfg.weights))
        print(f"weights from: {cfg.weights} are loaded.")

    if cfg.multigpu:
        model = DistributedDataParallel(model, device_ids=[device])
    # set optimizer, lr scheduler
    total_steps = len(train_dataset)
    optimizer = get_optimizer(model, cfg)
    # optimizer = Novograd(model.parameters(), cfg.lr)
    scheduler = get_scheduler(cfg, optimizer, total_steps, cfg.current_epoch)

    #seg_loss_func = DiceBceMultilabelLoss()
    #seg_loss_func = DiceLoss(sigmoid=True, smooth_nr=0.01, smooth_dr=0.01, include_background=True, batch=True)
    seg_loss_func = DiceFocalMultilabelLoss()
    #seg_loss_func = DiceMultilabelLoss()
    
    dice_metric = DiceMetric(reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    
    hausdorff_metric = HausdorffScore(reduction="mean")
    metric_function = [dice_metric, hausdorff_metric]
    metric_batch_function = dice_metric_batch


    post_pred = Compose([
        Activations(sigmoid=True),
        AsDiscrete(threshold=0.5),
    ])

    # train and val loop
    step = 0
    i = 0
    best_loss = 2
    if cfg.eval is True:
        best_val_metric = run_eval(
            model=model,
            val_dataloader=val_dataloader,
            post_pred=post_pred,
            metric_function=metric_function,
            metric_batch_function=metric_batch_function,
            seg_loss_func=seg_loss_func,
            cfg=cfg,
            epoch=0,
            device=device,
        )

    else:
        best_val_metric = 0.0
    best_weights_name = "best_weights"
    for epoch in range(cfg.current_epoch, cfg.epochs):
        cfg.current_epoch = epoch
        print("EPOCH:", epoch)
        if cfg.local_rank == 0 and cfg.usewandb:
            wandb.summary['current epoch'] = epoch
        gc.collect()
        if cfg.train is True:
            run_train(
                model=model,
                train_dataloader=train_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                seg_loss_func=seg_loss_func,
                cfg=cfg,
                # writer=writer,
                epoch=epoch,
                step=step,
                iteration=i,
                device = device
            )

            ##### for all data training
            if cfg.current_training_loss < best_loss:
                print(f"Found better loss: best loss {best_loss} -> {cfg.current_training_loss}")
                best_loss = cfg.current_training_loss


                if dist.get_rank() ==0:
                    torch.save(
                        model.module.state_dict(),
                        f"{cfg.output_dir}/fold{cfg.fold}/bestloss_{best_loss:.5}_epoch_{epoch}.pt",
                    )

                saved_weights = [os.path.join(f'{cfg.output_dir}/fold{cfg.fold}/', file) for file in os.listdir(f'{cfg.output_dir}/fold{cfg.fold}/')]
                if len(saved_weights) >= 10: #store 10 weights
                    oldest_file = min(saved_weights, key=os.path.getctime)
                    os.remove(oldest_file)

                if cfg.local_rank == 0 and cfg.usewandb:

                    wandb.summary['best_loss'] = best_loss
                    wandb.summary['best_metric_epoch'] = epoch

        if (epoch + 1) % cfg.eval_epochs == 0 and cfg.eval is True and epoch > cfg.start_eval_epoch:

            val_metric = run_eval(
                model=model,
                val_dataloader=val_dataloader,
                post_pred=post_pred,
                metric_function=metric_function,
                metric_batch_function=metric_batch_function,
                seg_loss_func=seg_loss_func,
                cfg=cfg,
                epoch=epoch,
                device=device,
            )
            if cfg.local_rank == 0 and cfg.usewandb:
                wandb.log({"val mean dice:": val_metric})


            
            if val_metric > best_val_metric:
                print(f"Find better metric: val_metric {best_val_metric:.5} -> {val_metric:.5}")
                best_val_metric = val_metric
                if dist.get_rank() ==0:
                    # checkpoint = create_checkpoint(
                    #     model,
                    #     optimizer,
                    #     epoch,
                    #     scheduler=scheduler,
                    # )
             
                    torch.save(
                        model.module.state_dict(),
                        f"{cfg.output_dir}/fold{cfg.fold}/dice_{best_val_metric:.5}_epoch_{epoch}_loss_{cfg.current_training_loss:.5}.pt",
                    )

                    saved_weights = [os.path.join(f'{cfg.output_dir}/fold{cfg.fold}/', file) for file in os.listdir(f'{cfg.output_dir}/fold{cfg.fold}/')]
                    if len(saved_weights) >= 10: #store 10 weights
                        oldest_file = min(saved_weights, key=os.path.getctime)
                        os.remove(oldest_file)
                    if cfg.local_rank == 0 and cfg.usewandb:

                        wandb.summary['best_metric'] = best_val_metric
                        wandb.summary['best_metric_epoch'] = epoch
                

    dist.destroy_process_group()
    if cfg.local_rank == 0 and cfg.usewandb:
        wandb.finish()       
            
def run_train(
    model,
    train_dataloader,
    optimizer,
    scheduler,
    seg_loss_func,
    cfg,
    # writer,
    epoch,
    step,
    iteration,
    device,
):
    model.train()
    scaler = GradScaler()
    progress_bar = tqdm(range(len(train_dataloader)))
    tr_it = iter(train_dataloader)
    dataset_size = 0
    running_loss = 0.0

    for itr in progress_bar:
        iteration += 1
        batch = next(tr_it)
        inputs, masks = (
            batch["image"].to(device),
            batch["mask"].to(device),
        )

        step += cfg.batch_size

        if cfg.amp is True:
            with autocast():
                outputs = model(inputs)
                loss = seg_loss_func(outputs, masks)
        else:
            outputs = model(inputs)
            loss = seg_loss_func(outputs, masks)
        if cfg.amp is True:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        optimizer.zero_grad()
        scheduler.step()
        
        running_loss += (loss.item() * cfg.batch_size)
        dataset_size += cfg.batch_size
        losses = running_loss / dataset_size
        progress_bar.set_description(f"loss: {losses:.4f} lr: {optimizer.param_groups[0]['lr']:.6f}")
        del batch, inputs, masks, outputs, loss
    print(f"Train loss: {losses:.4f}")
    cfg.current_training_loss = losses
    if cfg.local_rank == 0 and cfg.usewandb:
        wandb.log({"train loss": losses})
    torch.cuda.empty_cache()

def run_eval(model, val_dataloader, post_pred, metric_function, metric_batch_function, seg_loss_func, cfg, epoch, device):

    model.eval()

    dice_metric, hausdorff_metric = metric_function
    dice_metric_batch = metric_batch_function 

    progress_bar = tqdm(range(len(val_dataloader)))
    val_it = iter(val_dataloader)
    with torch.no_grad():
        for itr in progress_bar:
            batch = next(val_it)
            val_inputs, val_masks = (
                batch["image"].to(device),
                batch["mask"].to(device),
            )
            
            if cfg.val_amp is True:
                with autocast():
                    val_outputs = sliding_window_inference(val_inputs, cfg.roi_size, cfg.sw_batch_size, model)
            else:

                val_outputs = sliding_window_inference(val_inputs, cfg.roi_size, cfg.sw_batch_size, model, overlap=0.25)
                # print(val_outputs.shape, 'validation output before softmax')
                # val_outputs = nn.functional.softmax(val_outputs, dim=1)
                #print(val_outputs.shape, 'validation output shape')
                
            # cal metric
            # if cfg.run_tta_val is True:
            #     tta_ct = 1
            #     for dims in [[2],[3],[2,3]]:
            #         flip_val_outputs = sliding_window_inference(torch.flip(val_inputs, dims=dims), cfg.roi_size, cfg.sw_batch_size, model)
            #         val_outputs += torch.flip(flip_val_outputs, dims=dims)
            #         tta_ct += 1
                
            #     val_outputs /= tta_ct

            val_outputs = [post_pred(i) for i in val_outputs]
            val_outputs = torch.stack(val_outputs)
            #print(val_outputs.shape, 'val outputs shape')
            # metric is slice level put (n, c, h, w, d) to (n, d, c, h, w) to (n*d, c, h, w)
            # val_outputs = val_outputs.permute([0, 4, 1, 2, 3]).flatten(0, 1)
            # val_masks = val_masks.permute([0, 4, 1, 2, 3]).flatten(0, 1)

            #hausdorff_metric(y_pred=val_outputs, y=val_masks)
            dice_metric(y_pred=val_outputs, y=val_masks)
            dice_metric_batch(y_pred=val_outputs, y=val_masks)
    
            del val_outputs, val_inputs, val_masks, batch
    
    dice_batch = dice_metric_batch.aggregate()
    dice_score = dice_metric.aggregate().item()
    # print(f'lb_dice:{lb_dice.aggregate().item()}, sb_dice:{sb_dice.aggregate().item()}, st_dice:{st_dice.aggregate().item()}')
    #hausdorff_score = hausdorff_metric.aggregate().item()
    dice_metric.reset()
    dice_metric_batch.reset()
    #hausdorff_metric.reset()

    #all_score = dice_score * 0.4 + hausdorff_score * 0.6
    #wandb.log({'dice score': dice_score, "hausdorff score": hausdorff_score, "all score": all_score})
    #print(f"dice_score: {dice_score} hausdorff_score: {hausdorff_score} all_score: {all_score}")
    print(f'dice_score: {dice_score}')
    if cfg.local_rank == 0 and cfg.usewandb:
        wandb.log({'dice score': dice_score})
    print(dice_batch, 'dice batch')
    torch.cuda.empty_cache()

    #return all_score
    return dice_score



def main():
    parser = argparse.ArgumentParser(description="")
    sys.path.append("configs")
    parser.add_argument("-c", "--config", default="cfg_unet_multilabel", help="config filename")
    parser.add_argument("-f", "--fold", type=int, default=4, help="fold")
    parser.add_argument("--local_rank", type=int, help="node rank for distributed training")
    parser.add_argument("-s", "--seed", type=int, default=0, help="seed")
    parser.add_argument("-w", "--weights", default='/home/synergy/yhk/GI/output/all_dicefocal_epoch867_dataremoved/fold4/bestloss_0.071824_epoch_1108.pt', help="the path of weights")
    parser.add_argument("--multigpu", default=True)
    parser.add_argument("--usewandb", default=True)
    parser.add_argument("--output_dir", default="./output/1108finetune")
    parser_args, _ = parser.parse_known_args(sys.argv)

    cfg = importlib.import_module(parser_args.config).cfg
    cfg.fold = parser_args.fold
    cfg.seed = parser_args.seed
    cfg.weights = parser_args.weights
    cfg.local_rank = parser_args.local_rank
    cfg.multigpu = parser_args.multigpu
    cfg.usewandb = parser_args.usewandb
    cfg.output_dir = parser_args.output_dir
    

    if cfg.local_rank==0 and cfg.usewandb:
        wandb.init(project='uwgi_unet_ddp')
        wandb.run.name = cfg.output_dir.split('/')[-1]
        wandb.run.log_code(".")
        wandb.config.update(cfg)

    os.makedirs(str(cfg.output_dir + f"/fold{cfg.fold}/"), exist_ok=True)
    #wandb.config.update(cfg)
    main_worker(cfg)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    #wandb.init(entity='ykim102', project='uw-madison-gi-tract')
    #wandb.run.log_code(".")
    main()