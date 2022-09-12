"""
    IPA Datasets

    @author: Younghyun Kim
    Created: 2022.09.10
"""
import numpy as np

import torch
from torch.utils.data import Dataset


class IPADataset(Dataset):
    """
        IPA Dataset
    """
    def __init__(self, factors, target_weights):
        """
            Initialization

            Args:
                factors: multifactor scores of stocks
                    * dtype: np.array
                    * shape: (date_num, stock_num, factor_num)
                target_weights: target portfolio weights
                    * dtype: np.array
                    * shape: (strategy_num, date_num, stock_num)
        """
        self.date_num, self.stock_num, self.factor_num = factors.shape
        self.st_num = target_weights.shape[0]

        self.factors = factors
        self.target_weights = target_weights

        self.st_indices = torch.arange(self.st_num)

    def __len__(self):
        return self.date_num

    def __getitem__(self, idx):
        """
            get item

            Args:
                idx: index
            Returns:
                factors: multifactor scores
                    * dtype: torch.FloatTensor
                    * shape: (-1, strategy_num, stock_num, factor_num)
                target_weights: target portfolio weights
                    * dtype: torch.FloatTensor
                    * shape: (-1, strategy_num, stock_num)
                st_indices = strategy indices
                    * dtype: torch.LongTensor
                    * shape: (-1, strategy_num)
        """
        factors = torch.FloatTensor(
            self.factors[idx].astype(float)).unsqueeze(0)
        factors = factors.repeat(self.st_num, 1, 1)

        target_weights = torch.FloatTensor(
            self.target_weights[:, idx].astype(float))

        return factors, target_weights, self.st_indices