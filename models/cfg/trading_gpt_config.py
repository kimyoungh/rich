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

TRADING_GPT_CONFIG = {
    'factor_num': 8,
    'd_model': 32,
    'dim_ff': 32,
    'obs_map_nlayers': 1,
    'slope': 0.4,
    'dropout': 0.5,
    'nhead': 8,
    'nlayers': 12,
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
        MODEL_PATH, "trading_gpt_tf_rec/trading_gpt_tf_rec_best.pt"),
    'checkpoint_dir': "./ray_checkpoints/",
    'datasets_path':
        DATA_PATH+"trading_gpt/train_dataset_k200.pkl",
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
    'T_0': 5,
    'T_mult': 1,
    'num_workers': 2,
    'asset_idx': 0,
    'trading_period': 60,
    'trading_fee': 0.01,
    'alpha': 10,
    'beta': 1.,
    'gamma': 0.005,
    'delta': 20.,
    'iteration': 5,
    'initial_period': 3213,
    'test_trading_fee': 0.004,
}

_TRADER_BTC_TRAIN_CONFIG = {
    'best_sl_model_path': os.path.join(
        MODEL_PATH, "trading_gpt_tf_rec2/trading_gpt_tf_rec2_best.pt"),
    'datasets_path':
        DATA_PATH+"trading_gpt/train_dataset_btc.pkl",
    'asset_idx': 0,
    'trading_period': 60,
    'trading_fee': 0.001,
}

_TRADER_TRAIN_CL_CONFIG = {
    'train_period': 20,
    'train_window': 1250,
    'epoch_size': 10,
    'model_cl_name': 'trading_gpt_cl',
    'datasets_path':
        DATA_PATH+"trading_gpt/overall_dataset_k200.pkl",
    'best_model_path': None,
    'simulation_result_path': os.path.join(
        DATA_PATH, "simulations/"),
    'train_datasets': None,
    'rprob': 0.15,
}

TRADING_GPT_TRAIN_CONFIG = TRADING_GPT_CONFIG.copy()
TRADING_GPT_TRAIN_CONFIG.update(_TRADER_TRAIN_CONFIG)

TRADING_GPT_BTC_TRAIN_CONFIG = TRADING_GPT_TRAIN_CONFIG.copy()
TRADING_GPT_BTC_TRAIN_CONFIG.update(_TRADER_BTC_TRAIN_CONFIG)

TRADING_GPT_TRAIN_CL_CONFIG = TRADING_GPT_TRAIN_CONFIG.copy()
TRADING_GPT_TRAIN_CL_CONFIG.update(_TRADER_TRAIN_CL_CONFIG)