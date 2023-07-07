"""
    Configs for Picking Transformer with Cash

    @author: Younghyun Kim
    Created: 2023.02.11
"""
import os
import torch
from ray import tune

DATA_PATH = os.path.join(os.getcwd(), "train_data/")
MODEL_PATH = os.path.join(os.getcwd(), "trained/")

PICKING_TRANSFORMER2_CONFIG = {
    'seq_len': 30,
    'd_model': 32,
    'dim_ff': 64,
    'slope': 0.1,
    'dropout': 0.5,
    'nhead': 4,
    'nlayers': 4,
    'activation': 'gelu',
    'close_embeds_nlayers': 1,
    'value_time_embeds_nlayers': 1,
    'value_cs_embeds_nlayers': 1,
    'fusion_embeds_nlayers': 1,
    'allocator_map_nlayers': 1,
}

_TRADER_TRAIN_CONFIG = {
    'load_model': False,
    'load_model_path': None,
    'model_path': MODEL_PATH,
    'model_name': 'picking_transformer2',
    'checkpoint_dir': "./ray_checkpoints/",
    'close_data_path':
        DATA_PATH+"picking_transformer/close_data.npy",
    'value_data_path':
        DATA_PATH+"picking_transformer/value_data.npy",
    'returns_path':
        DATA_PATH+"picking_transformer/returns.npy",
    'epoch_size': 1000,
    'batch_size': 64,
    'lr': tune.grid_search(
        [0.001]),
    'amsgrad': True,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'valid_batch_size': 64,
    'valid_prob': 0.3,
    'asset_num_indices': [5, 20, 40, 50],
    'num_samples': 5,
    'device': 'cuda',
    'num_gpu': 1,
    'lr_scheduling': False,
    'sched_term': 5,
    'lr_decay': 0.99,
    'num_workers': 8,
    'data_num_workers': 2,
}

PICKING_TRANSFORMER2_TRAIN_CONFIG = PICKING_TRANSFORMER2_CONFIG.copy()
PICKING_TRANSFORMER2_TRAIN_CONFIG.update(_TRADER_TRAIN_CONFIG)