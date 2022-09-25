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
    def __init__(self, factors, target_weights, regimes):
        """
            Initialization

            Args:
                factors: multifactor scores of stocks
                    * dtype: np.array
                    * shape: (date_num, stock_num, factor_num)
                target_weights: target portfolio weights
                    * dtype: np.array
                    * shape: (strategy_num, date_num, stock_num)
                regimes: regime signal
                    * dtype: np.array
                    * shape: (date_num)
        """
        self.date_num, self.stock_num, self.factor_num = factors.shape
        self.st_num = target_weights.shape[0]

        self.factors = factors
        self.target_weights = target_weights
        self.regimes = regimes
        self.regime_categories = np.array(
            list(range(max(regimes) + 1)))

        self.st_indices = torch.arange(self.st_num)

        self.dates_list = np.array(list(range(self.date_num)))

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
                    * shape: (-1, strategy_numXregime_num, stock_num, factor_num)
                target_weights: target portfolio weights
                    * dtype: torch.FloatTensor
                    * shape: (-1, strategy_numXregime_num, stock_num)
                st_indices = strategy indices
                    * dtype: torch.LongTensor
                    * shape: (-1, strategy_numXregime_num)
        """
        reg_picked = self.regimes[idx]

        regs = np.setdiff1d(self.regime_categories, [reg_picked])

        factors_list, target_weights_list = [], []
        factors = torch.FloatTensor(
            self.factors[idx].astype(float)).unsqueeze(0)
        factors = factors.repeat(self.st_num, 1, 1)

        target_weights = torch.FloatTensor(
            self.target_weights[:, idx].astype(float))

        factors_list.append(factors)
        target_weights_list.append(target_weights)

        for reg in regs:
            ridx = np.random.choice(
                self.dates_list[self.regimes == reg], 1).item()

            factors = torch.FloatTensor(
                self.factors[ridx].astype(float)).unsqueeze(0)
            factors = factors.repeat(self.st_num, 1, 1)

            target_weights = torch.FloatTensor(
                self.target_weights[:, ridx].astype(float))

            factors_list.append(factors)
            target_weights_list.append(target_weights)

        factors = torch.cat(factors_list, dim=0)
        target_weights = torch.cat(target_weights_list, dim=0)

        return factors, target_weights, self.st_indices.repeat(
            len(self.regime_categories))