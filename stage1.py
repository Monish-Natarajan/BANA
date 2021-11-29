from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import os
import sys
from utils.common_util import seed,process_cfg,checkpoint_callback_stage1
from PytorchLightning.stage1.Lightningextension import VOCDataModule,LabelerLitModel
import wandb


wandb_logger = WandbLogger(project='BANA', # group runs in "BANA" project
                           log_model='all') # log all new checkpoints during training

def stage1(args):
    wandb.login()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    cfg=process_cfg(args.config_file)
    seed(cfg)
    datamodule=VOCDataModule(cfg)
    model=LabelerLitModel(cfg)
    wandb_logger.watch(model,log='all')  # logs histogram of gradients and parameters
    trainer = Trainer(max_epochs=cfg.SOLVER.MAX_ITER,logger=wandb_logger,callbacks=[checkpoint_callback_stage1(cfg)],gpus=[int(args.gpu_id)])
    trainer.fit(model,datamodule=datamodule)
    wandb.finish()
