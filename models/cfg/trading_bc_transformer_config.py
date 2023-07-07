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

TRADING_BC_TRANSFORMER_CONFIG = {
    'factor_num': 9,
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

    'reward_embeds_nlayers': 1,
    'action_map_nlayers': 1,
    'reward_map_nlayers': 1,
}

_TRADER_TRAIN_CONFIG = {
    'load_model': False,
    'load_model_path': None,
    'model_path': MODEL_PATH,
    'model_name': 'trading_bc_transformer',
    'checkpoint_dir': "./ray_checkpoints/",
    'datasets_path':
        DATA_PATH+"trading_bc_transformer/train_dataset_10.pkl",
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
}

TRADING_BC_TRANSFORMER_TRAIN_CONFIG = TRADING_BC_TRANSFORMER_CONFIG.copy()
TRADING_BC_TRANSFORMER_TRAIN_CONFIG.update(_TRADER_TRAIN_CONFIG)