# UW-Madison GI Tract Image Segmentation 70th Solution with 3D Unet

This repository contains training and inference code for kaggle competition: https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation.
3D dataset with 3D Unet were used for solution.

Datatset, training and testing pipeline were adopted from Yiheng Wang's kaggle rep.: https://www.kaggle.com/yiheng.


## Environment
- Python 3.8.13
- Pytorch 1.7.1+cu110
- Monai 0.8.1

## Notes
- Integrated distributed dataparallel for faster training. 
- Weight & Biases for logs .
- Training epochs 1000+ required for convergence.
- DynamicUnet, AttentionUnet, SwinUNETR, UNETR, BasicUNet did not perform as well as Unet with residual units.
- Bigger image sizes performed better (used 224, 224, 80).
- Due to limited number of submission and extended training period, I used all data instead of 5-folds. 
- For DDP training use: python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 ddp_train.py

## Thoughts
- 3D Unet with augmentations on CPU took about 2 days to train -> need to do augmentations on GPU. Applying elastic deformation was especially slow.
- Note other well performed strategies included ensemble of 2D, 2.5D, 3D datasets.

