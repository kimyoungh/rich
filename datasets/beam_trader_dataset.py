"""
    Beam Trader Datasets

    @author: Younghyun Kim
    Created: 2022.10.02
"""
import numpy as np

import torch
from torch.utils.data import Dataset


class BeamTraderDataset(Dataset):
    """
        Beam Trader Dataset
    """
    def __init__(self, factors, best_seqs, best_rebal_seqs):
        """
            Initialization

            Args:
                factors: multifactor scores of stocks
                    * dtype: np.array
                    * shape: (date_num, stock_num, factor_num)
                best_seqs: target best sequence
                    * dtype: np.array
                    * shape: (date_num, seq_len)
                best_rebal_seqs: target best rebal sequence
                    * dtype: np.array
                    * shape: (date_num, action_num, seq_len)
        """
        self.date_num, self.stock_num, self.factor_num = factors.shape
        _, self.action_num, self.seq_len = best_rebal_seqs.shape

        self.factors = factors
        self.best_seqs = best_seqs
        self.best_rebal_seqs = best_rebal_seqs

        self.data_length = self.date_num - 1

        self.dates_list = np.array(list(range(self.data_length)))

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        """
            get item

            Args:
                idx: index
            Returns:
                factors: multifactor scores
                    * dtype: torch.FloatTensor
                    * shape: (-1, stock_num, factor_num)
                factors_rebal: multifactor scores for rebal
                    * dtype: torch.FloatTensor
                    * shape: (-1, action_num, stock_num, factor_num)
                best_seqs: target best_sequence
                    * dtype: torch.LongTensor
                    * shape: (-1, seq_len)
                rebal_init_actions: target rebal init actions
                    * dtype: torch.LongTensor
                    * shape: (-1, action_num, 1)
                best_rebal_seqs: target best rebal sequence
                    * dtype: torch.LongTensor
                    * shape: (-1, action_num, seq_len-1)

        """
        factors_list, factors_rebal_list, best_seqs_list = [], [], []
        rebal_init_actions_list, best_rebal_seqs_list = [], []

        factors = torch.FloatTensor(
            self.factors[idx].astype(float))

        factors_rebal = torch.FloatTensor(
            self.factors[idx+1].astype(float)).unsqueeze(0)
        factors_rebal = factors_rebal.repeat(self.action_num, 1, 1)

        best_seqs = torch.LongTensor(
            self.best_seqs[idx].astype(int))

        rebal_seqs = torch.LongTensor(
            self.best_rebal_seqs[idx].astype(int))
        rebal_init_actions = rebal_seqs[:, 0].unsqueeze(-1)
        best_rebal_seqs = rebal_seqs[:, 1:]

        return factors, factors_rebal, best_seqs,\
            rebal_init_actions, best_rebal_seqs