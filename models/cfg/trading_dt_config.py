"""
    Configs for Trading GPT

    @author: Younghyun Kim
    Created: 2023.03.19
"""
import os
import torch
from ray import tune

DATA_PATH = os.path.join(os.getcwd(), "train_data/")
MODEL_PATH = os.path.join(os.getcwd(), "trained/")

TRADING_DT_CONFIG = {
    'factor_num': 8,
    'd_model': 32,
    'dim_ff': 32,
    'obs_map_nlayers': 1,
    'slope': 0.2,
    'dropout': 0.2,
    'nhead': 8,
    'nlayers': 8,
    'activation': 'gelu',
    'max_len': 5000,
    'actions': torch.arange(2),  # [Buy, Sell]

    'assets': {
        0: 'KODEX 200',
    },

    'reward_embeds_nlayers': 1,
    'action_map_nlayers': 1,
    'reward_map_nlayers': 1,
}

_TRADER_TRAIN_CONFIG = {
    'load_model': False,
    'load_model_path': None,
    'model_path': MODEL_PATH,
    'model_sl_name': 'trading_gpt_sl',
    'model_tf_name': 'trading_gpt_tf',
    'best_sl_model_path': os.path.join(
        MODEL_PATH, "trading_dt_sl/trading_dt_sl_best.pt"),
    'checkpoint_dir': "./ray_checkpoints/",
    'datasets_path':
        DATA_PATH+"trading_dt/train_dataset_k200.pkl",
    'epoch_size': 1000,
    'batch_size': 32,
    'train_num_workers': 8,
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
    'T_0': 10,
    'T_mult': 1,
    'num_workers': 2,
    'asset_idx': 0,
    'trading_period': 20,
    'trading_fee': 0.01,
    'alpha': 2.,
    'beta': 0.005,
    'iteration': 5,
    'initial_period': 250,
    'test_trading_fee': 0.004,
}

_TRADER_TRAIN_CL_CONFIG = {
    'epoch_size': 10,
    'model_cl_name': 'trading_gpt_cl',
    'datasets_path':
        DATA_PATH+"trading_gpt/overall_dataset_k200.pkl",
    'best_model_path': None,
    'simulation_result_path': os.path.join(
        DATA_PATH, "simulations/"),
    'train_datasets': None,
}

TRADING_DT_TRAIN_CONFIG = TRADING_DT_CONFIG.copy()
TRADING_DT_TRAIN_CONFIG.update(_TRADER_TRAIN_CONFIG)

TRADING_DT_TRAIN_CL_CONFIG = TRADING_DT_TRAIN_CONFIG.copy()
TRADING_DT_TRAIN_CL_CONFIG.update(_TRADER_TRAIN_CL_CONFIG)