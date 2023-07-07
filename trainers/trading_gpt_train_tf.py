"""
    Training procedure for TradingGPT(Target Finish)

    @author: Younghyun Kim
    Created: 2023.04.07
"""
import pickle
from copy import deepcopy
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

from models.cfg.trading_gpt_config import (TRADING_GPT_TRAIN_CONFIG,
                                        TRADING_GPT_BTC_TRAIN_CONFIG)
from datasets.trading_gpt_dataset import TradingGPTDataset
from models.trading_gpt import TradingGPT

def train_trader(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    train_loader, valid_loader = get_data_loaders(
        config['datasets_path'],
        config['trading_period'], config['valid_prob'],
        int(config['batch_size'] / config['num_workers']),
        int(config['valid_batch_size'] / config['num_workers']),
        config['num_workers'])

    model = TradingGPT(config).to(device)
    #model = torch.compile(model, backend='inductor')
    model.load_state_dict(
        torch.load(config['best_sl_model_path'],
                    map_location=device))
    model_sl = deepcopy(model)
    model.eval()
    model_sl.eval()

    model_sl.requires_grad_(False)

    optimizer = optim.Adam(model.parameters(),
                        lr=config['lr'],
                        amsgrad=config['amsgrad'],
                        betas=(config['beta_1'], config['beta_2']))

    lr_scheduling = config['lr_scheduling']

    if lr_scheduling:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, config['T_0'], config['T_mult'])
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

    assets = torch.tensor([config['asset_idx']])

    for epoch in range(config['epoch_size']):
        train(model, model_sl, optimizer, train_loader, assets,
            config['trading_fee'], config['alpha'], config['beta'],
            device)
        loss, loss_r, loss_k = validation(model, model_sl, valid_loader, assets,
                        config['trading_fee'], config['alpha'],
                        config['beta'], device)

        if lr_scheduling:
            scheduler.step()

        os.makedirs("model_trained", exist_ok=True)
        torch.save(model.state_dict(),
                "model_trained/checkpoint_model.pt")
        checkpoint = Checkpoint.from_directory("model_trained")

        session.report({"loss": loss,
                        "loss_r": loss_r,
                        "loss_k": loss_k},
                        checkpoint=checkpoint)

        if epoch % config['iteration'] == 0:
            model_sl = deepcopy(model)

def get_data_loaders(datasets_path, trading_period=60,
                    valid_prob=0.3, batch_size=64,
                    valid_batch_size=64, num_workers=4):
    """
        Get DataLoaders
    """
    with open(datasets_path, 'rb') as f:
        datasets = pickle.load(f)

    indices = np.arange(
        len(datasets['observations']) - trading_period + 1)
    valid_size = int(valid_prob * len(indices))
    valid_indices = np.random.choice(
        indices, valid_size, replace=False)

    train_indices = np.setdiff1d(indices, valid_indices)

    train_dataset = TradingGPTDataset(datasets, train_indices,
                                    trading_period)
    valid_dataset = TradingGPTDataset(datasets, valid_indices,
                                    trading_period)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=valid_batch_size,
        shuffle=True, num_workers=num_workers)

    return train_dataloader, valid_dataloader

def train(model, model_sl, optimizer, train_loader, assets,
        trading_fee, alpha, beta, device=None):
    """
        train method
    """
    device = device or torch.device('cpu')
    model.train()

    loss_kld = nn.KLDivLoss(reduction='batchmean')

    for batch, (obs, _, returns) in enumerate(train_loader):
        obs = obs.to(device)
        returns = returns.to(device)

        assets_in = assets.repeat(obs.shape[0]).to(device)

        acts_tf, rets_tf = get_trading_inference(
            model, assets_in, obs, returns, trading_fee)
        acts_sl, rets_sl = get_trading_inference(
            model_sl, assets_in, obs, returns, trading_fee)

        a_preds_tf, _ = model(assets_in, obs, acts_tf, rets_tf)
        a_preds_sl, _ = model_sl(assets_in, obs, acts_sl, rets_sl)

        actions_tf = torch.gather(
            a_preds_tf.softmax(-1), -1, acts_tf.unsqueeze(-1)).squeeze(-1)
        actions_sl = torch.gather(
            a_preds_sl.softmax(-1), -1, acts_sl.unsqueeze(-1)).squeeze(-1)

        returns_tf = actions_tf * rets_tf
        returns_sl = actions_sl * rets_sl

        loss_k = loss_kld(a_preds_tf.view(-1, 2).log_softmax(-1),
                        a_preds_sl.view(-1, 2).softmax(-1))
        loss_r = -torch.log((returns_tf - returns_sl).sigmoid()).mean()

        loss = (alpha * loss_r) + (beta * loss_k)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

@torch.no_grad()
def validation(model, model_sl, valid_loader, assets,
            trading_fee, alpha, beta, device=None):
    """
        validation method wow
    """
    device = device or torch.device('cpu')
    model.eval()

    loss_kld = nn.KLDivLoss(reduction='batchmean')

    losses, losses_k, losses_r = 0., 0., 0.

    length = len(valid_loader)

    for batch, (obs, _, returns) in enumerate(valid_loader):
        obs = obs.to(device)
        returns = returns.to(device)

        assets_in = assets.repeat(obs.shape[0]).to(device)

        acts_tf, rets_tf = get_trading_inference(
            model, assets_in, obs, returns, trading_fee)
        acts_sl, rets_sl = get_trading_inference(
            model_sl, assets_in, obs, returns, trading_fee)

        a_preds_tf, _ = model(assets_in, obs, acts_tf, rets_tf)
        a_preds_sl, _ = model_sl(assets_in, obs, acts_sl, rets_sl)

        actions_tf = torch.gather(
            a_preds_tf.softmax(-1), -1, acts_tf.unsqueeze(-1)).squeeze(-1)
        actions_sl = torch.gather(
            a_preds_sl.softmax(-1), -1, acts_sl.unsqueeze(-1)).squeeze(-1)

        returns_tf = actions_tf * rets_tf
        returns_sl = actions_sl * rets_sl

        loss_k = loss_kld(a_preds_tf.view(-1, 2).log_softmax(-1),
                        a_preds_sl.view(-1, 2).softmax(-1))

        loss_r = -torch.log((returns_tf - returns_sl).sigmoid()).mean()

        loss = (alpha * loss_r) + (beta * loss_k)

        losses += loss.item() / length
        losses_r -= loss_r.item() / length
        losses_k += loss_k.item() / length

    return losses, losses_r, losses_k

@torch.no_grad()
def get_trading_inference(model, assets, obs, returns, trading_fee):
    """
        get trading ineference

        Args:
            model
                * dtype: nn.Module
            assets
                * dtype: torch.LongTensor
                * shape: (batch_size)
            obs: observations
                * dtype: torch.FloatTensor
                * shape: (batch_size, trading_period, factor_num)
            returns: returns
                * dtype: torch.LongTensor
                * shape: (batch_size, trading_period)
            trading_fee
        Return:
            actions
                * dtype: torch.LongTensor
                * shape: (batch_size, trading_period)
            rets
                * dtype: torch.FloatTensor
                * shape: (batch_size, trading_period)
    """
    batch_size = obs.shape[0]
    trading_period = obs.shape[1]
    assert trading_period == returns.shape[1]

    actions, rets = (
        torch.tensor([]).type(torch.int).to(model.device),
        torch.tensor([]).to(model.device))
    for t in range(trading_period):
        if t == 0:
            action_preds, _ = model(assets, obs[:, :t+1])
        else:
            action_preds, _ = model(assets, obs[:, :t+1],
                                actions, rets)

        acts = action_preds.argmax(-1)
        acts_buy = (acts == 0).type(torch.float).view(-1)
        acts_sell = (acts == 1).type(torch.float).view(-1)

        if t > 0:
            tover = (actions[:, -1] != acts.view(-1)).type(torch.float)
        else:
            tover = 0.
        rs = ((returns[:, t] * acts_buy) - (returns[:, t] * acts_sell)
        ) - (2 * trading_fee * tover)

        actions = torch.cat((actions, acts), dim=1)
        rets = torch.cat((rets, rs.view(batch_size, 1)), dim=1)

    return actions, rets

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
        "--train_num_workers", type=int, default=8,
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
        "--model_tf_name", type=str, default="trading_gpt_tf",
        help="model tf name")
    parser.add_argument(
        "--device", type=str, default='cpu',
        help="device")
    parser.add_argument(
        "--lr", type=float, default=None,
        help="lr")
    parser.add_argument(
        "--lr_scheduling", action="store_true", default=False)
    parser.add_argument("--T_0", type=int, default=10)
    parser.add_argument("--T_mult", type=int, default=1)
    parser.add_argument('--train_btc', action="store_true",
                        default=False)

    args = parser.parse_args()

    return args

def main():
    """
        main training method
    """
    args = _parser()

    if args.train_btc:
        config = TRADING_GPT_BTC_TRAIN_CONFIG
    else:
        config = TRADING_GPT_TRAIN_CONFIG

    config['epoch_size'] = args.epoch_size
    config['batch_size'] = args.batch_size
    config['valid_batch_size'] = args.valid_batch_size
    config['model_tf_name'] = args.model_tf_name
    config['device'] = args.device
    config['num_samples'] = args.num_samples

    if args.lr is not None:
        config['lr'] = args.lr
    config['lr_scheduling'] = args.lr_scheduling
    config['train_num_workers'] = args.train_num_workers
    config['T_0'] = args.T_0
    config['T_mult'] = args.T_mult

    ray.init(num_cpus=config['train_num_workers'])

    sched = ASHAScheduler(max_t=config['epoch_size'])

    resources_per_trial = {
        "cpu": args.train_num_workers,
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
            name=config['model_tf_name'],
            local_dir=os.path.join(config['checkpoint_dir'],
                                config['model_tf_name']),
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
        config['model_path'], config['model_tf_name'],
    )
    if not os.path.exists(best_model_save_path):
        os.mkdir(best_model_save_path)

    # Save Best Model
    torch.save(best_model, os.path.join(
        best_model_save_path,
        config['model_tf_name']+"_best.pt"))

    print("Best config is: ", best_results.config)
    print(best_results.log_dir)


if __name__ == "__main__":
    main()