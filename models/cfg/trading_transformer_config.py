"""
    Configs for Trader

    @author: Younghyun Kim
    Created: 2022.10.15
"""
import os
import torch
from ray import tune

DATA_PATH = os.path.join(os.getcwd(), "train_data/")
MODEL_PATH = os.path.join(os.getcwd(), "trained/")

TRADING_TRANSFORMER_CONFIG = {
    'factor_num': 5,
    'd_model': 32,
    'dim_ff': 32,
    'asset_map_nlayers': 1,
    'slope': 0.2,
    'dropout': 0.3,
    'tt_nhead': 4,
    'tt_nlayers': 4,
    'activation': 'gelu',
    'recon_map_nlayers': 1,
    'task_map_nlayers': 1,
    'max_len': 5000,
    'actions': torch.arange(2),  # [Buy, Sell]
    ##  'rewards': torch.tensor([-10, 0, +1]),
    ##  'values': torch.arange(-100, 10+1),

    'assets': {
        0: 'KRW-XRP',
    },

    'value_embeds_nlayers': 1,
    'reward_embeds_nlayers': 1,
    'K': 50.,
    'value_map_nlayers': 1,
    'action_map_nlayers': 1,
    'reward_map_nlayers': 1,
}

_TRADER_TRAIN_CONFIG = {
    'load_model': False,
    'load_model_path': None,
    'model_path': MODEL_PATH,
    'model_name': 'trading_transformer',
    'checkpoint_dir': "./ray_checkpoints/",
    'datasets_path':
        DATA_PATH+"trading_transformer/train_dataset.pkl",
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

TRADING_TRANSFORMER_TRAIN_CONFIG = TRADING_TRANSFORMER_CONFIG.copy()
TRADING_TRANSFORMER_TRAIN_CONFIG.update(_TRADER_TRAIN_CONFIG)