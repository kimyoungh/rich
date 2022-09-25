"""
    Configs for IPA

    @author: Younghyun Kim
    Created: 2022.09.10
"""
import os

from ray import tune

DATA_PATH = os.path.join(os.getcwd(), "train_data/")
MODEL_PATH = os.path.join(os.getcwd(), "trained/")

IPA_CONFIG = {
    'factor_num': 69,
    'slope': 0.2,
    'dropout': 0.3,
    'd_model': 32,
    'nhead': 8,
    'nlayers': 8,
    'activation': 'gelu',
    'port_type_num': 100,
    'stock_embeds_map_nlayers': 3,
    'w_allocator_nlayers': 3,
}

IPA_TRAIN_CONFIG = {
    'factor_num': 69,
    'slope': 0.2,
    'dropout': 0.3,
    'd_model': 32,
    'nhead': 8,
    'nlayers': 8,
    'activation': 'gelu',
    'port_type_num': 100,
    'stock_embeds_map_nlayers': 3,
    'w_allocator_nlayers': 3,
    'load_model': False,
    'load_model_path': None,
    'model_path': MODEL_PATH,
    'model_name': 'ipa',
    'checkpoint_dir': "./ray_checkpoints/",
    'factors_path':
        DATA_PATH+"portfolio/factors_train.npy",
    'weights_path':
        DATA_PATH+"portfolio/weights_train.npy",
    'regimes_path':
        DATA_PATH+"portfolio/regimes_train.npy",
    'epoch_size': 1000,
    'batch_size': 256,
    'lr': tune.grid_search(
        [0.000001, 0.00001, 0.0001, 0.001, 0.01]),
    'amsgrad': True,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'valid_batch_size': 256,
    'valid_prob': 0.3,
    'num_samples': 5,
    'device': 'cuda',
    'num_gpu': 1,
    'lr_scheduling': False,
    'sched_term': 5,
    'lr_decay': 0.99,
    'num_workers': 4,
}