import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import os
import sys
import random
import argparse
import numpy as np
import torch

from configs.defaults import _C
from PytorchLightning.stage1.Lightningextension import VOCDataModule,LabelerLitModel
import wandb


wandb_logger = WandbLogger(project='BANA', # group runs in "BANA" project
                           log_model='all') # log all new checkpoints during training
checkpoint_callback = ModelCheckpoint(
    dirpath=f"{cfg.NAME}",    
    filename='{epoch}-{train_loss:.2f}',   # right now checking based on train_loss
    save_top_k =1,                 # saving best model, if to save the latest one replace by - save_last=True
    mode='min',                     # written for save_top_k
    every_n_epochs=40,              # after 40 epochs checkpoint saved.
    save_on_train_epoch_end=True   #  to run checkpointing at the end of the training epoch.  
    )     

def stage1(args):
    wandb.login()
    wandb_logger.watch(model,log='all')  # logs histogram of gradients and parameters
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    cfg=process_cfg(args.config_file)
    if cfg.SEED:  # need to see about this
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        random.seed(cfg.SEED)
        os.environ["PYTHONHASHSEED"] = str(cfg.SEED)

    datamodule=VOCDataModule(cfg)
    model=LabelerLitModel(cfg)
    trainer = Trainer(max_epochs=cfg.SOLVER.MAX_ITER,logger=wandb_logger,callbacks=[checkpoint_callback],gpus=[args.gpu_id])
    # load checkpoint 
    # need to see whether the function also works on .pt files
    model.backbone=model.backbone.load_from_checkpoint(f"./weights/{cfg.MODEL.WEIGHTS}")  # Just loading pre-trained weights
    trainer.fit(model,datamodule=datamodule)
    wandb.finish()

def process_cfg(config_file):
    cfg = _C.clone()
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg 