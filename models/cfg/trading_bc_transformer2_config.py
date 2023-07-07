"""
    Configs for Trading BC Transformer

    @author: Younghyun Kim
    Created: 2022.12.10
"""
import os
import torch
from ray import tune

DATA_PATH = os.path.join(os.getcwd(), "train_data/")
MODEL_PATH = os.path.join(os.getcwd(), "trained/")

TRADING_BC_TRANSFORMER2_CONFIG = {
    'factor_num': 8,
    'd_model': 64,
    'dim_ff': 64,
    'asset_map_nlayers': 1,
    'slope': 0.2,
    'dropout': 0.3,
    'tt_nhead': 4,
    'tt_nlayers': 4,
    'activation': 'gelu',
    'max_len': 5000,
    'actions': torch.arange(2),  # [Buy, Sell]

    'assets': {
        0: 'KRW-XRP',
    },
    'action_map_nlayers': 1,
}

_TRADER_TRAIN_CONFIG = {
    'load_model': False,
    'load_model_path': None,
    'model_path': MODEL_PATH,
    'model_name': 'trading_bc_transformer2',
    'checkpoint_dir': "./ray_checkpoints/",
    'datasets_path':
        DATA_PATH+"trading_bc_transformer/train_dataset_5.pkl",
    'epoch_size': 1000,
    'batch_size': 32,
    'lr': tune.grid_search(
        [0.001]),
    'amsgrad': True,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'valid_batch_size': 32,
    'valid_prob': 0.3,
    'num_samples': 5,
    'device': 'cpu',
    'num_gpu': 1,
    'lr_scheduling': False,
    'sched_term': 5,
    'lr_decay': 0.99,
    'num_workers': 8,
    'asset_idx': 0,
    'pretrain_epoch_prob': 0.5,
    'updown_coeff': 0.5,
}

_TRADER_TRAIN_K200_CONFIG = {
    'datasets_path':
        DATA_PATH+"trading_bc_transformer/train_dataset_20_k200.pkl",
}

TRADING_BC_TRANSFORMER2_TRAIN_CONFIG = TRADING_BC_TRANSFORMER2_CONFIG.copy()
TRADING_BC_TRANSFORMER2_TRAIN_CONFIG.update(_TRADER_TRAIN_CONFIG)
TRADING_BC_TRANSFORMER2_TRAIN_K200_CONFIG =\
    TRADING_BC_TRANSFORMER2_TRAIN_CONFIG.copy()
TRADING_BC_TRANSFORMER2_TRAIN_K200_CONFIG.update(_TRADER_TRAIN_K200_CONFIG)