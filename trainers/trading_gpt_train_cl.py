"""
    Training procedure for TradingGPT with Continual Learning

    @author: Younghyun Kim
    Created: 2023.04.23
"""
import pickle
from copy import deepcopy
import argparse
import os
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from models.cfg.trading_gpt_config import TRADING_GPT_TRAIN_CL_CONFIG
from datasets.trading_gpt_dataset import TradingGPTDataset
from models.trading_gpt import TradingGPT
from utils.target_finish_utils import *

def train_trader(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('train_trader')

    train_loader = get_data_loaders(
        config['train_datasets'],
        config['trading_period'],
        config['batch_size'],
        config['num_workers'])

    model = TradingGPT(config).to(device)

    if config['best_model_path'] is not None:
        model.load_state_dict(
            torch.load(
                os.path.join(config['best_model_path'],
                config['model_cl_name']+"_best.pt"),
                map_location=device))

    #model = torch.compile(model, backend='inductor')
    model_prev = deepcopy(model)
    model.eval()
    model_prev.eval()

    model_prev.requires_grad_(False)

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

    assets = torch.tensor([config['asset_idx']])

    for epoch in range(config['epoch_size']):
        train(model, model_prev, optimizer, train_loader, assets,
            config['trading_fee'], config['alpha'], config['beta'],
            config['gamma'], config['delta'], config['rprob'], device)

        if lr_scheduling:
            scheduler.step()

        if epoch % config['iteration'] == 0:
            model_prev = deepcopy(model)
            model_prev.requires_grad_(False)
            model_prev.eval()

    return model

def get_data_loaders(datasets, trading_period=60,
                    batch_size=64, num_workers=4):
    """
        Get DataLoaders
    """
    train_indices = np.arange(
        len(datasets['observations']) - trading_period + 1)

    train_dataset = TradingGPTDataset(datasets, train_indices,
                                    trading_period)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers)

    return train_dataloader

def train(model, model_prev, optimizer, train_loader, assets,
        trading_fee, alpha, beta, gamma, delta, rprob, device=None):
    """
        train method
    """
    device = device or torch.device('cpu')
    model.train()

    loss_kld = nn.KLDivLoss(reduction='batchmean')

    alpha = torch.FloatTensor([float(alpha)]).to(device)
    beta = torch.FloatTensor([float(beta)]).to(device)
    gamma_pt = torch.FloatTensor([float(gamma)]).to(device)

    for batch, (obs, _, returns) in enumerate(train_loader):
        obs = obs.to(device)
        returns = returns.to(device)

        assets_in = assets.repeat(obs.shape[0]).to(device)

        acts_tf, rets_tf = get_trading_inference(
            model, assets_in, obs, returns, trading_fee, rprob)
        acts_sl, rets_sl = get_trading_inference(
            model_prev, assets_in, obs, returns, trading_fee, rprob)

        a_preds_tf, _ = model(assets_in, obs, acts_tf, rets_tf)
        a_preds_sl, _ = model_prev(assets_in, obs, acts_sl, rets_sl)

        actions_tf = torch.gather(
            a_preds_tf.softmax(-1), -1, acts_tf.unsqueeze(-1)).squeeze(-1)
        actions_sl = torch.gather(
            a_preds_sl.softmax(-1), -1, acts_sl.unsqueeze(-1)).squeeze(-1)

        loss_avg_r = avg_return_tf(actions_tf, actions_sl,
                                rets_tf, rets_sl, alpha)
        loss_cagr = cagr_tf(actions_tf, actions_sl,
                            rets_tf, rets_sl, alpha)
        loss_to = turnover_tf(a_preds_tf, a_preds_sl, gamma_pt)
        loss_vol = volatility_tf(actions_tf, actions_sl,
                                rets_tf, rets_sl, beta)
        loss_mdd = mdd_tf(actions_tf, actions_sl,
                        rets_tf, rets_sl, beta)
        loss_k = loss_kld(a_preds_tf.view(-1, 2).log_softmax(-1),
                        a_preds_sl.view(-1, 2).softmax(-1))
        loss_direc = direction_tf(a_preds_tf, returns, delta)

        loss = loss_avg_r + loss_cagr + loss_to + loss_vol +\
            loss_mdd + (gamma * loss_k) + loss_direc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

@torch.no_grad()
def get_trading_inference(model, assets, obs, returns, trading_fee,
                        rprob=0.15):
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
            rprob: random probability
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

        alen = action_preds.shape[-1]
        acts_rep = action_preds.argmax(-1)

        if np.random.rand() <= rprob:
            acts = torch.randint_like(acts_rep, 0, alen)
        else:
            acts = acts_rep
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
        "--epoch_size", type=int, default=10,
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
        "--model_cl_name", type=str, default="trading_gpt_cl",
        help="model cl name")
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

def train_process(config: dict = None):
    """
        train process
    """
    model = train_trader(config)

    best_model_save_path = os.path.join(
        config['model_path'], config['model_cl_name'],
    )
    if not os.path.exists(best_model_save_path):
        os.mkdir(best_model_save_path)
    print(best_model_save_path)

    # Save Best Model
    torch.save(model.state_dict(), os.path.join(
        best_model_save_path,
        config['model_cl_name']+"_best.pt"))

    # Save Best Model with Serial Number
    torch.save(model.state_dict(), os.path.join(
        best_model_save_path,
        config['model_cl_name']+"_"+config['train_date']+"_best.pt"))
    print(config['train_date'])

    config['best_model_path'] = best_model_save_path

def simulation(config: dict = None):
    """
        simulation
    """
    trading_period = config['trading_period']
    trading_fee = config['test_trading_fee']

    with open(config['datasets_path'], 'rb') as f:
        datasets = pickle.load(f)

    observations = datasets['observations']
    returns = datasets['returns']
    timestamps = datasets['timestamps']

    train_window = config['train_window']

    assert len(observations) > config['initial_period']
    date_length = len(observations) - config['initial_period']
    init_period = config['initial_period']

    obs = torch.FloatTensor(observations.astype(float)).unsqueeze(0)
    rets = torch.FloatTensor(
        returns.astype(float)).unsqueeze(0).unsqueeze(-1)

    assets = torch.LongTensor([0])
    actions, rets_series = (
        torch.tensor([]).type(torch.int),
        torch.tensor([]))

    action_log, rets_log = [], []

    if not os.path.exists(config['simulation_result_path']):
        os.makedirs(config['simulation_result_path'], exist_ok=True)

    print("Simulation starts....")

    for t in range(date_length):
        time_t = init_period + t
        if t % config['train_period'] == 0:
            if t == 0:
                config['train_datasets'] = {
                'observations': observations[:time_t],
                'returns': returns[:time_t],
                'timestamps':timestamps[:time_t],
                }
            else:
                config['train_datasets'] = {
                    'observations': observations[
                        time_t-train_window:time_t],
                    'returns': returns[time_t-train_window:time_t],
                    'timestamps': timestamps[time_t-train_window:time_t],
                }
            config['train_date'] = timestamps[time_t]
            train_process(config)

            model = TradingGPT(config)
            model.eval()
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        config['best_model_path'],
                        config['model_cl_name']+"_best.pt"),
                    map_location='cpu'))

        with torch.no_grad():
            if t == 0:
                action_preds, _ = model(assets,
                                        obs[:, init_period:time_t+1])
            elif t < trading_period:
                action_preds, _ = model(assets,
                                        obs[:, init_period:time_t+1],
                                        actions, rets_series)
            else:
                action_preds, _ = model(
                    assets,
                    obs[:, time_t-trading_period+1:time_t+1],
                    actions, rets_series)

        acts = action_preds.argmax(-1)
        acts_buy = (acts == 0).type(torch.float).view(-1)
        acts_sell = (acts == 1).type(torch.float).view(-1)

        if t > 0:
            tover = (actions[:, -1] != acts.view(-1)).type(torch.float)
        else:
            tover = 0.
        rs = ((rets[:, time_t, 0] * (acts_buy - acts_sell))
            - (2 * trading_fee * tover))

        actions = torch.cat((actions, acts), dim=1)
        rets_series = torch.cat((rets_series, rs.view(1, 1)), dim=1)

        if t >= trading_period - 1:
            actions = actions[:, 1:]
            rets_series = rets_series[:, 1:]

        action_log.append(acts.item())
        rets_log.append(rs.item())

    action_log = np.array(action_log)
    rets_log = np.array(rets_log)

    rst = pd.DataFrame(np.stack((action_log, rets_log), axis=-1),
                    columns=['actions', 'model_returns'],
                    index=timestamps[init_period:])
    rst['bm_returns'] = returns[init_period:]

    rst.to_parquet(os.path.join(
        config['simulation_result_path'],
        config['model_cl_name']+"_simulation_result.pq"))

def main():
    """
        main training method
    """
    args = _parser()

    config = TRADING_GPT_TRAIN_CL_CONFIG

    config['epoch_size'] = args.epoch_size
    config['batch_size'] = args.batch_size
    config['valid_batch_size'] = args.valid_batch_size
    config['model_cl_name'] = args.model_cl_name
    config['device'] = args.device
    config['num_samples'] = args.num_samples

    if args.lr is not None:
        config['lr'] = args.lr
    config['lr_scheduling'] = args.lr_scheduling
    config['train_num_workers'] = args.train_num_workers
    config['T_0'] = args.T_0
    config['T_mult'] = args.T_mult

    simulation(config)

if __name__ == "__main__":
    main()