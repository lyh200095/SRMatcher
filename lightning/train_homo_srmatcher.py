import os
import torch
from pytorch_lightning.plugins.precision.deepspeed import DeepSpeedPrecisionPlugin

import math

import sys
import argparse
import pprint
from distutils.util import strtobool
from pathlib import Path
from loguru import logger as loguru_logger
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy

from model.loftr_src.lightning.data import HomoDataModule
from model.loftr_src.config.default import get_cfg_defaults
from model.loftr_src.utils.misc import get_rank_zero_only_logger, setup_gpus
from model.loftr_src.utils.profiler import build_profiler
from lightning.lightning_homo_srmatcher import PL_H_SRMatcher

loguru_logger = get_rank_zero_only_logger(loguru_logger)

def parse_args():
    # init a costum parser which will be added into pl.MyTrainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--data_cfg_path', type=str, help='data loftr_config path', default='./train_config/homo_trainval_640.py')
    parser.add_argument(
        '--loftr_cfg_path', type=str, help='loftr loftr_config path', default='./train_config/loftr_ds_dense.py')
    parser.add_argument(
        '--exp_name', type=str, default='default_exp_name')
    parser.add_argument(
        '--batch_size', type=int, default=1, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=2)
    parser.add_argument(
        '--pin_memory', type=lambda x: bool(strtobool(x)),
        nargs='?', default=True, help='whether loading data to pinned memory or not')
    parser.add_argument(
        '--ckpt_path', type=str, default=None,
        help='pretrained checkpoint path, helpful for using a pre-trained coarse-only LoFTR')
    parser.add_argument(
        '--disable_ckpt', action='store_true',
        help='disable checkpoint saving (useful for debugging).')
    parser.add_argument(
        '--profiler_name', type=str, default=None,
        help='options: [inference, pytorch], or leave it unset')
    parser.add_argument(
        '--parallel_load_data', action='store_true',
        help='load datasets in with multiple processes.')

    return parser.parse_args()

import torch

torch.set_float32_matmul_precision('high')


def main():
    args = parse_args()
    args.num_nodes = 1
    args.gpus = os.environ.get('CUDA_VISIBLE_DEVICES', "1,2")
    rank_zero_only(pprint.pprint)(vars(args))

    # init default-cfg and merge it with the main- and data-cfg
    default_config = get_cfg_defaults()
    # model_config = get_cfg_model()
    default_config.merge_from_file(args.loftr_cfg_path)
    default_config.merge_from_file(args.data_cfg_path)
    # pl.seed_everything(default_config.TRAINER.SEED)  # reproducibility
    # TODO: Use different seeds for each dataloader workers
    # This is needed for data augmentation

    # scale lr and warmup-step automatically
    args.gpus = _n_gpus = setup_gpus(args.gpus)
    default_config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
    default_config.TRAINER.TRUE_BATCH_SIZE = default_config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = default_config.TRAINER.TRUE_BATCH_SIZE / default_config.TRAINER.CANONICAL_BS
    default_config.TRAINER.SCALING = _scaling
    default_config.TRAINER.TRUE_LR = default_config.TRAINER.CANONICAL_LR * _scaling * 1  # set 0.5 to finetune
    default_config.TRAINER.WARMUP_STEP = math.floor(default_config.TRAINER.WARMUP_STEP / _scaling)
    # lightning module
    profiler = build_profiler(args.profiler_name)


    loguru_logger.info(f"LightningModule initialized!")

    # lightning data
    data_module = HomoDataModule(args, default_config)
    loguru_logger.info(f"DataModule initialized!")

    # TensorBoard Logger
    logger = TensorBoardLogger(
        save_dir='logs/tb_logs',
        name=args.exp_name,
        version='ablation_dino_s',
        default_hp_metric=False)

    cktp_dir = str(Path(logger.log_dir) / 'checkpoints')
    ckpt_callback = ModelCheckpoint(
        monitor='val_auc@3px',
        verbose=True,
        save_top_k=5,
        mode='max',
        save_last=True,
        dirpath=cktp_dir,
        filename='{val_loss:.3f}-{epoch:02d}-{val_auc@3px:.4f}'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor, ckpt_callback] if not args.disable_ckpt else [lr_monitor]

    os.makedirs(logger.log_dir, exist_ok=True)
    model = PL_H_SRMatcher(default_config, profiler=profiler, pretrained_ckpt=args.ckpt_path, log_dir=logger.log_dir)

    # Lightning MyTrainer
    trainer = pl.Trainer(
        accelerator="gpu",
        precision=32,
        strategy=DDPStrategy(find_unused_parameters=True,),
        max_epochs=60,
        num_sanity_val_steps=0,
        num_nodes=1,
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=not args.disable_ckpt,
        enable_progress_bar=True,
        sync_batchnorm=False,
        gradient_clip_val=default_config.TRAINER.GRADIENT_CLIPPING,
        profiler=profiler,
        log_every_n_steps=25,
        check_val_every_n_epoch=1,
    )

    loguru_logger.info(f"Trainer initialized!")
    loguru_logger.info(f"Start training!")
    trainer.fit(model, datamodule=data_module, ckpt_path=('last' if os.path.exists(os.path.join(cktp_dir, 'last.ckpt')) else None))

    # ckpt_path='logs/tb_logs/default_exp_name/ablation_dino_s/checkpoints/val_loss=2.872-epoch=26-val_auc@3px=0.7136.ckpt'
    # trainer.validate(model, datamodule=data_module, ckpt_path=ckpt_path)

if __name__ == '__main__':
    main()

