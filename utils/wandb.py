import wandb
import os
import numpy as np


def init_wandb(model, cfg) -> None:
    """
    Initialize project on Weights & Biases
    Args:
        model (Torch Model): Model for Training
        args (TrainOptions,optional): TrainOptions class (refer options/train_options.py). Defaults to None.
    """
    
    wandb.init(
        name=cfg.NAME,
        config=cfg,
        project="BANA",
    )

    wandb.watch(model, log="all")


def wandb_log(train_loss, lr, iter):
    """
    Logs the accuracy and loss to wandb
    Args:
        train_loss (float): Training loss
        val_loss (float): Validation loss
        train_acc (float): Training Accuracy
        val_acc (float): Validation Accuracy
        epoch (int): Epoch Number
    """

    wandb.log({
        'Loss': train_loss,
        'Learning Rate': lr,
    }, step=iter)