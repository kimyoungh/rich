"""
    Trainer for IPA

    @author: Younghyun Kim
    Created: 2022.09.10
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
from ray.tune.schedulers import ASHAScheduler

from models.cfg.ipa_config import IPA_TRAIN_CONFIG
from datasets.ipa_dataset import IPADataset
from models.portfolio import InvestingPortfolioAllocator


class TrainIPA(tune.Trainable):
    """
        Trainer for IPA
    """
    def setup(self, config):
        """
            setup
        """
        use_cuda = config.get("device") == "cuda" and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.train_loader, self.valid_loader = self.get_data_loaders(
            config['factors_path'], config['weights_path'],
            config['valid_prob'], config['batch_size'],
            config['valid_batch_size'])

        self.model = InvestingPortfolioAllocator(config).to(self.device)
        self.model.eval()
        print("setup")

        if config['load_model']:
            self.model.load_state_dict(
                torch.load(config['load_model_path'],
                        map_location=self.device))

        self.optimizer = optim.Adam(self.model.parameters(),
                            lr=config['lr'],
                            amsgrad=config['amsgrad'],
                            betas=(config['beta_1'], config['beta_2']))

        self.lr_scheduling = config['lr_scheduling']

        if self.lr_scheduling:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, config['sched_term'],
                gamma=config['lr_decay'])
        else:
            self.scheduler = None

        self.loss_fn = nn.KLDivLoss(reduction='batchmean')

    def step(self):
        print("step")
        self.train()
        loss = self.validation()

        if self.lr_scheduling:
            self.scheduler.step()

        return {"loss": loss}

    def get_data_loaders(self, factors_path, weights_path,
                        valid_prob=0.3, batch_size=64,
                        valid_batch_size=64):
        """
            Get DataLoaders
        """
        factors = np.load(factors_path, allow_pickle=True)
        weights = np.load(weights_path, allow_pickle=True)

        indices = np.arange(factors.shape[0])
        valid_size = int(valid_prob * len(indices))
        valid_indices = np.random.choice(
            indices, valid_size, replace=False)

        train_indices = np.setdiff1d(indices, valid_indices)

        train_dataset = IPADataset(factors[train_indices],
                                weights[:, train_indices])
        valid_dataset = IPADataset(factors[valid_indices],
                                weights[:, valid_indices])

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=valid_batch_size, shuffle=True)

        return train_dataloader, valid_dataloader

    def train(self):
        """
            train method
        """
        self.model.train()

        for (factors, target_weights, st_indices) in self.train_loader:
            factors = factors.view(-1, factors.shape[-2],
                                factors.shape[-1]).to(self.device)
            target_weights = target_weights.view(
                -1, factors.shape[-2]).to(self.device)
            st_indices = st_indices.view(-1).to(self.device)

            w_preds, _ = self.model(factors, st_indices)
            loss = self.loss_fn(w_preds.log(), target_weights)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def validation(self):
        """
            validation method
        """
        self.model.eval()

        losses = 0.

        for batch, (factors,
                    target_weights, st_indices) in self.valid_loader:
            factors = factors.view(-1, factors.shape[-2],
                                factors.shape[-1]).to(self.device)
            target_weights = target_weights.view(
                -1, factors.shape[-2]).to(self.device)
            st_indices = st_indices.view(-1).to(self.device)

            with torch.no_grad():
                w_preds, _ = self.model(factors, st_indices)
                loss = self.loss_fn(w_preds.log(), target_weights)

            losses += loss.item() / (batch + 1)

        return losses

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
        torch.save(self.model.state_dict(), checkpoint_path)

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path,
                                            map_location=self.device))


def main(epoch_size=None, device=None):
    """
        main training method
    """
    args = _parser()

    config = IPA_TRAIN_CONFIG

    if epoch_size is not None:
        config['epoch_size'] = epoch_size
    else:
        config['epoch_size'] = args.epoch_size
    config['batch_size'] = args.batch_size
    config['valid_batch_size'] = args.valid_batch_size
    config['model_name'] = args.model_name

    if device is not None:
        config['device'] = device
    else:
        config['device'] = args.device
    config['num_samples'] = args.num_samples

    if args.lr is not None:
        config['lr'] = args.lr
    config['lr_scheduling'] = args.lr_scheduling
    config['sched_term'] = args.sched_term
    config['lr_decay'] = args.lr_decay

    ray.init(num_cpus=args.num_workers)
    sched = ASHAScheduler()

    tuner = tune.Tuner(
        tune.with_resources(TrainIPA, resources={
            "cpu": args.num_workers, "gpu": config['num_gpu']
                if torch.cuda.is_available() else 0}),
        run_config=air.RunConfig(
            name=config['model_name'],
            local_dir=os.path.join(config['checkpoint_dir'],
                                config['model_name']),
            stop={"training_iteration": config['epoch_size']},
            checkpoint_config=air.CheckpointConfig(
                checkpoint_at_end=True, checkpoint_frequency=3
            )),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode='min',
            scheduler=sched,
            num_samples=config['num_samples']),
        param_space=config)

    results = tuner.fit()

    print("Best_config is: ", results.get_best_result().config)

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


if __name__ == "__main__":
    main()