import os
import sys
#from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from utils.common_util import seed,process_cfg
from PytorchLightning.stage2.Lightningextension import VOCDataModule,LightningModel

#logger = logging.getLogger("stage2")

def stage2(cfg):
    wandb.login()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    cfg=process_cfg(args.config_file)
    seed(cfg)
    datamodule=VOCDataModule(cfg)
    model=LightningModel(cfg)
    # logger.info(f"START {cfg.NAME} -->")
    # wandb_logger.watch(model,log='all')  # logs histogram of gradients and parameters
    trainer = Trainer(gpus=[int(args.gpu_id)])
    # logger.info(f"END {cfg.NAME} -->") 

