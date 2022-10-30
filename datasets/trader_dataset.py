"""
    Trader Datasets

    @author: Younghyun Kim
    Created: 2022.10.17
"""
import numpy as np

import torch
from torch.utils.data import Dataset


class TraderDataset(Dataset):
    """
        Trader Dataset
    """
    def __init__(self, factors, gfactors, weights,
                return_series, indices, rand_prob=0.2):
        """
            Initialization

            Args:
                factors: multifactor scores of stocks
                    * dtype: np.array
                    * shape: (date_num, stock_num, factor_num)
                gfactors: multifactor scores of additional stocks
                    * dtype: np.array
                    * shape: (date_num, add_stock_num, factor_num)
                weights: portfolio weights
                    * dtype: np.array
                    * shape: (date_num, stock_num)
                return_series: return series
                    * dtype: np.array
                    * shape: (date_num, seq_len, stock_num)
                indices: date point index
                    * dtype: np.array
                    * shape: (index_num)
                rand_prob: picking probability of random weights
                    * default: 0.2
        """
        self.date_num, self.stock_num, self.factor_num = factors.shape
        _, self.seq_len, _ = return_series.shape

        self.factors = factors
        self.gfactors = gfactors
        self.weights = weights
        self.return_series = return_series
        self.indices = indices
        self.rand_prob = rand_prob

        self.index_num = len(indices)

    def __len__(self):
        return self.index_num

    def __getitem__(self, idx):
        """
            get item

            Args:
                idx: index
            Returns:
                factors: multifactor scores
                    * dtype: torch.FloatTensor
                    * shape: (-1, stock_num, factor_num)
                gfactors: additional multifactor scores
                    * dtype: torch.FloatTensor
                    * shape: (-1, add_stock_num, factor_num)
                weights: portfolio weights
                    * dtype: torch.FloatTensor
                    * shape: (-1, stock_num)
                weights_rec: recent portfolio weights
                    * dtype: torch.FloatTensor
                    * shape: (-1, stock_num)
                returns: return series
                    * dtype: torch.FloatTensor
                    * shape: (-1, seq_len, stock_num)
        """
        ind = self.indices[idx]

        factors = torch.FloatTensor(
            self.factors[ind].astype(float))

        gfactors = torch.FloatTensor(
            self.gfactors[ind].astype(float))

        weights = torch.FloatTensor(
            self.weights[ind].astype(float))

        returns = self.return_series[ind]

        if np.random.rand() <= self.rand_prob or ind == 0:
            weights_rec = torch.rand_like(weights)
            weights_rec = weights_rec / weights_rec.sum(dim=-1).item()
        else:
            weights_rec = torch.FloatTensor(
                self.weights[ind-1].astype(float))

        return factors, gfactors, weights, weights_rec, returns