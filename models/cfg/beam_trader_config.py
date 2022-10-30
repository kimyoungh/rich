"""
    Configs for Beam Trader

    @author: Younghyun Kim
    Created: 2022.09.25
"""
import os

from ray import tune

DATA_PATH = os.path.join(os.getcwd(), "train_data/")
MODEL_PATH = os.path.join(os.getcwd(), "trained/")

BEAM_TRADER_CONFIG = {
    'factor_num': 69,
    'action_num': 5,
    'max_len': 100,
    'slope': 0.1,
    'dropout': 0.1,
    'd_model': 16,
    'nhead': 1,
    'nlayers': 1,
    'activation': 'gelu',
    'obs_embeds_map_nlayers': 1,
    'action_preds_map_nlayers': 1,
}

_BEAM_TRADER_TRAIN_CONFIG = {
    'd_model': 16,
    'nhead': 1,
    'nlayers': 1,
    'obs_embeds_map_nlayers': 1,
    'load_model': False,
    'load_model_path': None,
    'model_path': MODEL_PATH,
    'model_name': 'beam_trader',
    'checkpoint_dir': "./ray_checkpoints/",
    'factors_path':
        DATA_PATH+"beam_trader/factors_train.npy",
    'gfactors_path':
        DATA_PATH+"beam_trader/gfactors_train.npy",
    'best_seqs_path':
        DATA_PATH+"beam_trader/best_seqs_train.npy",
    'best_rebal_seqs_path':
        DATA_PATH+"beam_trader/best_rebal_seqs_train.npy",
    'epoch_size': 100,
    'batch_size': 32,
    'lr': tune.grid_search([0.001, 0.01, 0.1]),
    'amsgrad': True,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'valid_batch_size': 32,
    'valid_prob': 0.2,
    'num_samples': 2,
    'device': 'cuda',
    'num_gpu': 1,
    'lr_scheduling': False,
    'sched_term': 5,
    'lr_decay': 0.99,
    'num_workers': 8,
}

BEAM_TRADER_TRAIN_CONFIG = BEAM_TRADER_CONFIG.copy()
BEAM_TRADER_TRAIN_CONFIG.update(_BEAM_TRADER_TRAIN_CONFIG)