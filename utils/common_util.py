import numpy as np
import random
import torch
from configs.defaults import _C
from pytorch_lightning.callbacks import ModelCheckpoint

def seed(cfg):
    if cfg.SEED:  # need to see about this
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        random.seed(cfg.SEED)
        os.environ["PYTHONHASHSEED"] = str(cfg.SEED)

def process_cfg(config_file):
    cfg = _C.clone()
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file")
    parser.add_argument("--stage", type=str, default="1", help="select stage")
    parser.add_argument("--gpu-id", type=str, default="0", help="select a GPU index")
    return parser.parse_args()

def checkpoint_callback_stage1(cfg):
    return ModelCheckpoint(
    dirpath=f"{cfg.NAME}",    
    filename='{epoch}-{train_loss:.2f}',   # right now checking based on train_loss
    save_top_k =1,                 # saving best model, if to save the latest one replace by - save_last=True
    mode='min',                     # written for save_top_k
    every_n_epochs=40,              # after 40 epochs checkpoint saved.
    save_on_train_epoch_end=True   #  to run checkpointing at the end of the training epoch.  
    )