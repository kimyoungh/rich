"""
    Training procedure for IPA

    @author: Younghyun Kim
    Created: 2022.09.12
"""
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

from models.cfg.ipa_config import IPA_TRAIN_CONFIG
from datasets.ipa_dataset import IPADataset
from models.portfolio import InvestingPortfolioAllocator

def train_ipa(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    train_loader, valid_loader = get_data_loaders(
        config['factors_path'], config['weights_path'],
        config['regimes_path'],
        config['valid_prob'], config['batch_size'],
        config['valid_batch_size'], config['num_workers'])

    model = InvestingPortfolioAllocator(config).to(device)
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

    for _ in range(config['epoch_size']):
        train(model, optimizer,train_loader, device)
        loss = validation(model, valid_loader, device)

        if lr_scheduling:
            scheduler.step()

        os.makedirs("model_trained", exist_ok=True)
        torch.save(model.state_dict(),
                "model_trained/checkpoint_model.pt")
        checkpoint = Checkpoint.from_directory("model_trained")

        session.report({"loss": loss}, checkpoint=checkpoint)

def get_data_loaders(factors_path, weights_path, regimes_path,
                    valid_prob=0.3, batch_size=64,
                    valid_batch_size=64, num_workers=4):
    """
        Get DataLoaders
    """
    factors = np.load(factors_path, allow_pickle=True)
    weights = np.load(weights_path, allow_pickle=True)
    regimes = np.load(regimes_path, allow_pickle=True)

    indices = np.arange(factors.shape[0])
    valid_size = int(valid_prob * len(indices))
    valid_indices = np.random.choice(
        indices, valid_size, replace=False)

    train_indices = np.setdiff1d(indices, valid_indices)

    train_dataset = IPADataset(factors[train_indices],
                            weights[:, train_indices],
                            regimes[train_indices])
    valid_dataset = IPADataset(factors[valid_indices],
                            weights[:, valid_indices],
                            regimes[valid_indices])

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

    loss_fn = nn.KLDivLoss(reduction='batchmean')

    for (factors, target_weights, st_indices) in train_loader:
        factors = factors.view(-1, factors.shape[-2],
                            factors.shape[-1]).to(device)
        target_weights = target_weights.view(
            -1, factors.shape[-2]).to(device)
        st_indices = st_indices.view(-1).to(device)

        w_preds, _ = model(factors, st_indices)
        loss = loss_fn(w_preds.log(), target_weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validation(model, valid_loader, device=None):
    """
        validation method
    """
    device = device or torch.device('cpu')
    model.eval()

    loss_fn = nn.KLDivLoss(reduction='batchmean')

    losses = 0.

    for batch, (factors,
                target_weights, st_indices) in enumerate(valid_loader):
        factors = factors.view(-1, factors.shape[-2],
                            factors.shape[-1]).to(device)
        target_weights = target_weights.view(
            -1, factors.shape[-2]).to(device)
        st_indices = st_indices.view(-1).to(device)

        with torch.no_grad():
            w_preds, _ = model(factors, st_indices)
            loss = loss_fn(w_preds.log(), target_weights)

        losses += loss.item() / (batch + 1)

    return losses

def _parser():
    """
        argparse parser method

        Return:
            args
    """
    parser = argparse.ArgumentParser(
        description="argument parser for training CrossAssetBERT"
    )

    parser.add_argument(
        "--num_workers", type=int, default=4,
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
        "--checkpoint_epoch", type=int, default=3,
        help="checkpoint epoch")
    parser.add_argument(
        "--num_samples", type=int, default=5,
        help="num samples from ray tune")
    parser.add_argument(
        "--model_name", type=str, default="ipa",
        help="model name")
    parser.add_argument(
        "--device", type=str, default='cuda',
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

    config = IPA_TRAIN_CONFIG

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

    ray.init(num_cpus=config['num_workers'])

    sched = ASHAScheduler(max_t=config['epoch_size'])

    resources_per_trial = {
        "cpu": args.num_workers,
        "gpu": 1 if config['device'] == 'cuda'
                and torch.cuda.is_available() else 0}
    tuner = tune.Tuner(
        tune.with_resources(train_ipa, resources=resources_per_trial),
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
                            map_location=config['device'])

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