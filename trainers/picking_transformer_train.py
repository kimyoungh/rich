"""
    Training procedure for Picking Transformer

    @author: Younghyun Kim
    Created: 2023.01.29
"""
import pickle
import argparse
import os
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import ray
from ray import air, tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler

from models.cfg.picking_transformer_config import PICKING_TRANSFORMER_TRAIN_CONFIG
from datasets.picking_transformer_dataset import PickingTransformerDataset
from models.picking_transformer import PickingTransformer

def train_trader(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    train_loader, valid_loader = get_data_loaders(
        config['close_data_path'], config['value_data_path'],
        config['returns_path'], config['seq_len'],
        config['valid_prob'],
        int(config['batch_size'] / config['num_workers']),
        int(config['valid_batch_size'] / config['num_workers']),
        config['asset_num_indices'],
        config['data_num_workers'])

    model = PickingTransformer(config).to(device)
    model.eval()

    optimizer = optim.Adam(model.parameters(),
                        lr=config['lr'],
                        amsgrad=config['amsgrad'],
                        betas=(config['beta_1'], config['beta_2']))

    lr_scheduling = config['lr_scheduling']

    if lr_scheduling:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, config['sched_term'],
            gamma=config['lr_decay'])
    else:
        scheduler = None

    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt"),
                map_location=device)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    for epoch in range(config['epoch_size']):
        train(model, optimizer, train_loader, device)
        loss, loss_max, loss_prets = validation(
            model, valid_loader, device)

        if lr_scheduling:
            scheduler.step()

        os.makedirs("model_trained", exist_ok=True)
        torch.save(model.state_dict(),
                "model_trained/checkpoint_model.pt")
        checkpoint = Checkpoint.from_directory("model_trained")

        session.report({"loss": loss,
                        "loss_max": loss_max,
                        "loss_prets": loss_prets},
                        checkpoint=checkpoint)

def get_data_loaders(close_data_path, value_data_path,
                    returns_path, window=30,
                    valid_prob=0.3, batch_size=64,
                    valid_batch_size=64,
                    asset_num_indices=[5, 20, 40, 50],
                    num_workers=4):
    """
        Get DataLoaders
    """
    close_data = np.load(close_data_path, allow_pickle=True)
    value_data = np.load(value_data_path, allow_pickle=True)
    returns = np.load(returns_path, allow_pickle=True)

    indices = np.arange(close_data.shape[0])
    indices = indices[window-1:-1]

    index_indices = np.arange(len(indices))

    valid_size = int(valid_prob * len(indices))
    valid_indices = np.random.choice(
        index_indices, valid_size, replace=False)

    train_indices = np.setdiff1d(index_indices, valid_indices)

    train_dataset = PickingTransformerDataset(
        close_data, value_data, returns, indices[train_indices],
        asset_num_indices=asset_num_indices,
        window=window)
    valid_dataset = PickingTransformerDataset(
        close_data, value_data, returns, indices[valid_indices],
        asset_num_indices=asset_num_indices,
        window=window)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=valid_batch_size,
        shuffle=True, num_workers=num_workers)

    return train_dataloader, valid_dataloader

def train(model, optimizer, train_loader, device=None):
    """
        train method
    """
    device = device or torch.device('cpu')
    model.train()

    loss_kld = nn.KLDivLoss(reduction='batchmean')

    for batch, (
        obs_dict, max_weights_dict, rets_dict
        ) in enumerate(train_loader):

        for num in obs_dict.keys():
            obs = obs_dict[num].to(device)
            max_weights = max_weights_dict[num].to(device)
            rets = rets_dict[num].to(device)

            # Max Return Weights
            preds, _ = model(obs, softmax=True)
            loss = loss_kld(preds.log(), max_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # port rets
            preds, _ = model(obs, softmax=True)
            loss = -(preds * rets).sum(dim=-1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

@torch.no_grad()
def validation(model, valid_loader, device=None):
    """
        validation method
    """
    device = device or torch.device('cpu')
    model.eval()

    loss_kld = nn.KLDivLoss(reduction='batchmean')

    losses = 0.
    losses_max, losses_prets = 0., 0.

    for batch, (
        obs_dict, max_weights_dict, rets_dict
        ) in enumerate(valid_loader):

        num_len = len(obs_dict)

        for num in obs_dict.keys():
            obs = obs_dict[num].to(device)
            max_weights = max_weights_dict[num].to(device)
            rets = rets_dict[num].to(device)

            # Max Return Weights
            preds, _ = model(obs, softmax=True)
            loss_max = loss_kld(preds.log(), max_weights)

            # Return Upgrade
            preds, _ = model(obs, softmax=True)
            loss_prets = (preds * rets).sum(dim=-1).mean()

            losses += (
                loss_max.item() - loss_prets.item())/ 2. / num_len
            losses_max += loss_max.item() / num_len
            losses_prets += loss_prets.item() / num_len

    losses /= (batch + 1)
    losses_max /= (batch + 1)
    losses_prets /= (batch + 1)

    return losses, losses_max, losses_prets

def _parser():
    """
        argparse parser method

        Return:
            args
    """
    parser = argparse.ArgumentParser(
        description="argument parser for training Trader"
    )

    parser.add_argument(
        "--num_workers", type=int, default=8,
        help="Set number of workers for training")
    parser.add_argument(
        "--data_num_workers", type=int, default=2,
        help="Set number of workers for dataloader")
    parser.add_argument(
        "--epoch_size", type=int, default=1000,
        help="epoch size")
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="batch size")
    parser.add_argument(
        "--valid_batch_size", type=int, default=32,
        help="valid batch size")
    parser.add_argument(
        "--checkpoint_epoch", type=int, default=1,
        help="checkpoint epoch")
    parser.add_argument(
        "--num_samples", type=int, default=5,
        help="num samples from ray tune")
    parser.add_argument(
        "--model_name", type=str, default="picking_transformer",
        help="model name")
    parser.add_argument(
        "--device", type=str, default='cpu',
        help="device")
    parser.add_argument(
        "--lr", type=float, default=None,
        help="lr")
    parser.add_argument(
        "--lr_scheduling", action="store_true", default=False)
    parser.add_argument(
        "--sched_term", type=int, default=5)
    parser.add_argument(
        "--lr_decay", type=float, default=0.99)

    args = parser.parse_args()

    return args

def main():
    """
        main training method
    """
    args = _parser()

    config = PICKING_TRANSFORMER_TRAIN_CONFIG

    config['epoch_size'] = args.epoch_size
    config['batch_size'] = args.batch_size
    config['valid_batch_size'] = args.valid_batch_size
    config['model_name'] = args.model_name
    config['device'] = args.device
    config['num_samples'] = args.num_samples

    if args.lr is not None:
        config['lr'] = args.lr
    config['lr_scheduling'] = args.lr_scheduling
    config['sched_term'] = args.sched_term
    config['lr_decay'] = args.lr_decay
    config['num_workers'] = args.num_workers
    config['data_num_workers'] = args.data_num_workers

    ray.init(num_cpus=config['num_workers'])

    sched = ASHAScheduler(max_t=config['epoch_size'])

    resources_per_trial = {
        "cpu": args.num_workers,
        "gpu": 1 if config['device'] == 'cuda'
                and torch.cuda.is_available() else 0}
    tuner = tune.Tuner(
        tune.with_resources(train_trader,
                            resources=resources_per_trial),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=sched,
            num_samples=args.num_samples
        ),
        run_config=air.RunConfig(
            name=config['model_name'],
            local_dir=os.path.join(config['checkpoint_dir'],
                                config['model_name']),
        ),
        param_space=config
    )
    results = tuner.fit()
    best_results = results.get_best_result()
    best_model_path = os.path.join(
        str(best_results.log_dir), "model_trained/checkpoint_model.pt")

    best_model = torch.load(best_model_path,
                            map_location=torch.device(config['device']))

    best_model_save_path = os.path.join(
        config['model_path'], config['model_name'],
    )
    if not os.path.exists(best_model_save_path):
        os.mkdir(best_model_save_path)

    # Save Best Model
    torch.save(best_model, os.path.join(
        best_model_save_path,
        config['model_name']+"_best.pt"))

    print("Best config is: ", best_results.config)
    print(best_results.log_dir)


if __name__ == "__main__":
    main()