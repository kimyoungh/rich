"""
    Training procedure for Trading BC Transformer 2

    @author: Younghyun Kim
    Created: 2022.12.18
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

from models.cfg.trading_bc_transformer2_config import (
    TRADING_BC_TRANSFORMER2_TRAIN_CONFIG,
    TRADING_BC_TRANSFORMER2_TRAIN_K200_CONFIG)
from datasets.trading_bc_transformer2_dataset import TradingBCTransformer2Dataset
from models.trading_bc_transformer2 import TradingBCTransformer2

def train_trader(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    train_loader, valid_loader = get_data_loaders(
        config['datasets_path'], config['asset_idx'],
        config['valid_prob'],
        int(config['batch_size'] / config['num_workers']),
        int(config['valid_batch_size'] / config['num_workers']),
        config['num_workers'])

    model = TradingBCTransformer2(config).to(device)
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

    pretrain_epoch = int(
        config['pretrain_epoch_prob'] * config['epoch_size'])

    for epoch in range(config['epoch_size']):
        if epoch < (pretrain_epoch - 1):
            pretrain = True
        else:
            pretrain = False

        train(model, optimizer, train_loader, device,
            pretrain, config['updown_coeff'])
        loss, loss_ud, loss_act = validation(
            model, valid_loader, device, config['updown_coeff'])

        if lr_scheduling:
            scheduler.step()

        os.makedirs("model_trained", exist_ok=True)
        torch.save(model.state_dict(),
                "model_trained/checkpoint_model.pt")
        checkpoint = Checkpoint.from_directory("model_trained")

        session.report({"loss": loss,
                        "loss_ud": loss_ud,
                        "loss_act": loss_act},
                        checkpoint=checkpoint)

def get_data_loaders(datasets_path, asset_idx=0,
                    valid_prob=0.3, batch_size=64,
                    valid_batch_size=64, num_workers=4):
    """
        Get DataLoaders
    """
    with open(datasets_path, 'rb') as f:
        datasets = pickle.load(f)

    observations = datasets['observations']
    action_series = datasets['action_series']
    updown_series = datasets['updown_series']

    indices = np.arange(observations.shape[0])
    valid_size = int(valid_prob * len(indices))
    valid_indices = np.random.choice(
        indices, valid_size, replace=False)

    train_indices = np.setdiff1d(indices, valid_indices)

    train_dataset = TradingBCTransformer2Dataset(
        observations[train_indices], action_series[train_indices],
        updown_series[train_indices], asset_idx)
    valid_dataset = TradingBCTransformer2Dataset(
        observations[valid_indices], action_series[valid_indices],
        updown_series[valid_indices], asset_idx)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=valid_batch_size,
        shuffle=True, num_workers=num_workers)

    return train_dataloader, valid_dataloader

def train(model, optimizer, train_loader, device=None, pretrain=False,
        updown_coeff=0.2):
    """
        train method
    """
    device = device or torch.device('cpu')
    model.train()

    loss_ca = nn.CrossEntropyLoss()

    for batch, (
        assets_in, obs, actions, updowns) in enumerate(train_loader):
        assets_in = assets_in.to(device)
        obs = obs.to(device)
        actions = actions.to(device)
        updowns = updowns.to(device)

        if pretrain:
            a_preds, _ = model(assets_in, obs, updowns)

            loss = loss_ca(a_preds.view(-1, a_preds.shape[-1]),
                            updowns.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            a_preds, _ = model(assets_in, obs, actions)

            loss = loss_ca(a_preds.view(-1, a_preds.shape[-1]),
                            actions.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            a_preds, _ = model(assets_in, obs, updowns)

            loss = loss_ca(a_preds.view(-1, a_preds.shape[-1]),
                            updowns.view(-1)) * updown_coeff
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

@torch.no_grad()
def validation(model, valid_loader, device=None,
            pretrain=False, updown_coeff=0.2):
    """
        validation method
    """
    device = device or torch.device('cpu')
    model.eval()

    loss_ca = nn.CrossEntropyLoss()

    losses, losses_ud, losses_act = 0., 0., 0.

    for batch, (
        assets_in, obs, actions, updowns) in enumerate(valid_loader):
        assets_in = assets_in.to(device)
        obs = obs.to(device)
        actions = actions.to(device)
        updowns = updowns.to(device)

        a_preds, _ = model(assets_in, obs, updowns)
        loss_ud = loss_ca(a_preds.view(-1, a_preds.shape[-1]),
                        updowns.view(-1))

        a_preds, _ = model(assets_in, obs, actions)
        loss_act = loss_ca(a_preds.view(-1, a_preds.shape[-1]),
                        actions.view(-1))

        losses_ud += loss_ud.item() / (batch + 1)
        losses_act += loss_act.item() / (batch + 1)

        if pretrain:
            losses = losses_ud
        else:
            losses += (
                (loss_ud.item() * updown_coeff)
                + loss_act.item()) / (2 * (batch + 1))

    return losses, losses_ud, losses_act

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
        "--model_name", type=str, default="trading_bc_transformer2",
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
    parser.add_argument(
        "--k200", action="store_true", default=False)
    parser.add_argument(
        "--pretrain_epoch_prob", type=float, default=0.5,
        help="pretrain epoch prob")
    parser.add_argument(
        "--updown_coeff", type=float, default=0.5,
        help="updown coeff")

    args = parser.parse_args()

    return args

def main():
    """
        main training method
    """
    args = _parser()

    if args.k200:
        config = TRADING_BC_TRANSFORMER2_TRAIN_K200_CONFIG
    else:
        config = TRADING_BC_TRANSFORMER2_TRAIN_CONFIG

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
    config['pretrain_epoch_prob'] = args.pretrain_epoch_prob
    config['updown_coeff'] = args.updown_coeff

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