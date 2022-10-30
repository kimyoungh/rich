"""
    Configs for Trader

    @author: Younghyun Kim
    Created: 2022.10.15
"""
import os

from ray import tune

DATA_PATH = os.path.join(os.getcwd(), "train_data/")
MODEL_PATH = os.path.join(os.getcwd(), "trained/")

TRADER_CONFIG = {
    'factor_num': 69,
    'add_factor_num': 69,
    'slope': 0.1,
    'dropout': 0.1,
    'd_model': 32,
    'nhead': 1,
    'nlayers': 2,
    'activation': 'gelu',
    'factor_embeds_map_nlayers': 1,
    'w_allocator_nlayers': 2,
    'asset_list': {
        0: 'K200',
        1: 'KQ',
        2: 'K200_i',
        3: 'KQ_i',
        4: 'K200_2X',
        5: 'KQ_2X',
        6: 'K200_i_2X',
    },
}

_TRADER_TRAIN_CONFIG = {
    'load_model': False,
    'load_model_path': None,
    'model_path': MODEL_PATH,
    'model_name': 'trader',
    'checkpoint_dir': "./ray_checkpoints/",
    'factors_path':
        DATA_PATH+"trader/factors_train.npy",
    'gfactors_path':
        DATA_PATH+"trader/gfactors_train.npy",
    'weights_path':
        DATA_PATH+"trader/weights_train.npy",
    'return_series_path':
        DATA_PATH+"trader/return_series_train.npy",
    'rand_prob': 0.2,
    'fee': 0.001,
    'epoch_size': 1000,
    'batch_size': 32,
    'lr': tune.grid_search(
        [0.0001, 0.001]),
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
}

TRADER_TRAIN_CONFIG = TRADER_CONFIG.copy()
TRADER_TRAIN_CONFIG.update(_TRADER_TRAIN_CONFIG)