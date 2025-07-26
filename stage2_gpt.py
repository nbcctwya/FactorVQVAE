import copy
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from pathlib import Path
from typing import Union, Callable, Optional
import wandb
import os
import qlib
from qlib.contrib.data.handler import Alpha158
import pickle
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import load_yaml_param_settings, load_args, get_root_dir, save_model, seed_everything, run_inference, UnfreezeDecoderCallback
from trainer.autoregressive import minGPT
import logging
from qlib.data.dataset import DatasetH, TSDatasetH, DataHandlerLP
from data.dataset import init_data_loader
import time
torch.set_float32_matmul_precision('high')

def train_stage2(config, train_loader, valid_loader, test_loader):
    
    tf_hidden = config['transformer']['hidden_size']
    tf_head = config['transformer']['heads']
    tf_layers = config['transformer']['n_layers']
    seed = config['train']['seed']
    vq_hidden = config['vqvae']['hidden_size']
    vq_elements = config['vqvae']['num_elements']
    vq_code = config['vqvae']['num_factors']
    alpha = config['vqvae']['alpha']
    rank_alpha = config['transformer']['rank_loss_alpha']
    project_name = config['train']['project_name']
    dec_warmup = config['transformer']['dec_warmup']

    if config['train']['run_name'] is not None:
        run_name = f'Stage2_VQ{vq_code}_Th{tf_hidden}_h{tf_head}_l{tf_layers}_sd{seed}' # !Auto
    else:
        raise NotImplementedError("run_name should be specified. We recommend to use the same run_name as stage1.")

    # * Init model
    n_train_samples = len(train_loader) * config['train']['batch_size'] # approximate

    model = minGPT(config=config, n_train_samples=n_train_samples)
    
    #* init logger
    group_name = config['train']['group_name'] if config['train']['group_name'] is not None else "실험 중"
    wandb.init(project=project_name, name=run_name, config=config, group= group_name,entity="x7jeon8gi") # todo: group_name
    wandb_logger = WandbLogger(project=project_name, name=run_name, config=config,entity="x7jeon8gi")
    wandb_logger.watch(model, log='all')

    chekcpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='Val_RIC',
        mode='max',
        dirpath=os.path.join(get_root_dir(), 'checkpoints'),
        filename = f'{run_name}'+'-{epoch}-{val_loss:.4f}'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=15, # epochs to wait after min has been reached
        verbose=True,
        mode='min'
    )

    callbacks =[LearningRateMonitor(logging_interval='step'),
                chekcpoint_callback, 
                early_stop_callback]
        
    trainer = pl.Trainer(logger = wandb_logger,
                    enable_checkpointing=True,
                    callbacks=callbacks,
                    max_epochs=config['train']['num_epochs'],
                    accelerator= 'gpu', # 'gpu' # ! 디버깅을 위해 device를 cpu로 설정
                    # strategy='ddp',
                    devices= 1, # config['train']['gpu_counts'] if torch.cuda.is_available() else None,
                    num_nodes=1,
                    precision = config['train']['precision'],
                    gradient_clip_val=3.0
                    )
    
    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = valid_loader)
    # Best Model Load
    model = minGPT.load_from_checkpoint(chekcpoint_callback.best_model_path, config=config, n_train_samples=n_train_samples)
    model.eval()
    # run inference
    pred_df, rank_ic, metric = run_inference(model, test_loader)
    pred_df.to_pickle(f"{get_root_dir()}/res/{run_name}.pkl")

    # log metric
    wandb.log(metric)

    logging.info("Saving Models.")
    save_model({'maskgit': model.mingpt})

    wandb.finish()

if __name__ =="__main__":
    #* Load config
    args = load_args()
    config = load_yaml_param_settings(args.config)
    seed_everything(config['train']['seed'])

    #* Load dataset
    #* Load dataset
    pickle_path = config['data']['data_path']
    prefix = config['data']['prefix']
    universe = config['data']['universe']
    window_size = config['data']['window_size']
    if pickle_path and os.path.exists(pickle_path):
        print(f"========== Loading data from pickle: {pickle_path} ==========")

        with open(f'{pickle_path}/{prefix}/{universe}_others_{window_size}_dl_train.pkl', 'rb') as f:
            train_prepare = pickle.load(f)
        with open(f'{pickle_path}/{prefix}/{universe}_others_{window_size}_dl_valid.pkl', 'rb') as f:
            valid_prepare = pickle.load(f)
        with open(f'{pickle_path}/{prefix}/{universe}_others_{window_size}_dl_test.pkl', 'rb') as f:
            test_prepare = pickle.load(f)
    
    else:
        print(f"Using Alpha158 handler with qlib data")
        data_handler_config = load_yaml_param_settings(config['data']['data_handler_config'])
        qlib.init(provider_uri=config['data'].get('provider_uri', "./qlib_data/cn_data"), region=data_handler_config['region'])
        dataset = Alpha158(**data_handler_config)

        segments = {
            'train': config['data']['train_period'],
            'valid': config['data']['valid_period'],
            'test': config['data']['test_period'],
        }

        TsDataset = TSDatasetH(
            handler=dataset, 
            segments=segments, 
            step_len=config['data']['window_size'], 
        )

        train_prepare = TsDataset.prepare(segments='train', data_key=DataHandlerLP.DK_L)
        valid_prepare = TsDataset.prepare(segments='valid', data_key=DataHandlerLP.DK_L)
        test_prepare = TsDataset.prepare(segments='test', data_key=DataHandlerLP.DK_I)
        train_prepare.config(fillna_type='ffill+bfill')
        valid_prepare.config(fillna_type='ffill+bfill')
        test_prepare.config(fillna_type='ffill+bfill')

    train_loader = init_data_loader(train_prepare, shuffle=True, num_workers=config['train']['num_workers'])
    valid_loader = init_data_loader(valid_prepare, shuffle=False, num_workers=config['train']['num_workers'])
    test_loader = init_data_loader(test_prepare, shuffle=False, num_workers=config['train']['num_workers'])

    #* Train
    start_time = time.time()
    train_stage2(config, train_loader, valid_loader, test_loader)
    end_time = time.time()

    logging.info(f"Training time: {end_time - start_time} seconds.")