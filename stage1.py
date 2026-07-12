import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pickle
import qlib
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import TSDatasetH, DataHandlerLP
from trainer.autoencoder import FactorVQVAE
import os
from utils import load_yaml_param_settings, load_args, get_root_dir, seed_everything
from data.dataset import init_data_loader
torch.set_float32_matmul_precision('high')

def train(config, train_prepare, valid_prepare):
    codebook_sizes = config['vqvae']['num_factors']
    hidden_size = config['vqvae']['hidden_size']
    elements = config['vqvae']['num_elements']
    alpha = config['vqvae']['alpha']
    project_name = config['train']['project_name']
    seed = config['train']['stage1_seed']
    market = config['experiment']['market']
    if config['train']['run_name'] is not None:
        run_name = f'{market}_Stage1_VQ{codebook_sizes}_sd{seed}'
    else:
        run_name = None

    #* Init model
    n_train_samples = len(train_loader.dataset)
    
    model = FactorVQVAE(config, n_train_samples, ckpt_path=None, ignore_keys=list())

    #* Init logger
    tensorboard_logger = TensorBoardLogger(
        save_dir=config['paths']['log_dir'],
        name=project_name,
        version=run_name
    )

    chekcpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor=config['train']['stage1_monitor'],
        mode='min',
        dirpath=config['paths']['checkpoint_dir'],
        filename='stage1_best'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=config['train']['stage1_early_stop'],
        verbose=True,
        mode='min'
    )

    trainer = pl.Trainer(logger = tensorboard_logger,
                         enable_checkpointing=True,
                         callbacks=[LearningRateMonitor(logging_interval='step'), 
                                    chekcpoint_callback, 
                                    early_stop_callback],
                         max_epochs=config['train']['num_epochs'],
                         accelerator=config['train']['device'],
                         # strategy='ddp',
                         devices=config['train']['devices'],
                         precision = config['train']['precision'],
                         )

    trainer.fit(model, train_dataloaders = train_prepare, val_dataloaders = valid_prepare)


if __name__ == "__main__":

    #* Load config
    args = load_args()
    config = load_yaml_param_settings(args.config)
    
    # * Set seed
    seed_everything(config['train']['stage1_seed'])

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

    num_workers = config['train']['num_workers']
    train_loader = init_data_loader(train_prepare, shuffle=True, num_workers=config['train']['num_workers'])
    valid_loader = init_data_loader(valid_prepare, shuffle=False, num_workers=config['train']['num_workers'])

    train(config, train_loader, valid_loader)
