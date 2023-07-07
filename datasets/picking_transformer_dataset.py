"""
    Picking Transformer Datasets

    @author: Younghyun Kim
    Created: 2023.01.29
"""
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset


class PickingTransformerDataset(Dataset):
    """
        Picking Transformer Dataset
    """
    def __init__(self, close_data, value_data, returns, indices,
                asset_num_indices=[5, 20, 40, 50], window=30, eps=1e-6):
        """
            Initialization

            Args:
                close_data
                    * dtype: np.array
                    * shape: (date_num, asset_num)
                value_data
                    * dtype: np.array
                    * shape: (date_num, asset_num)
                returns
                    * dtype: np.array
                    * shape: (date_num, asset_num)
                indices:
                    * dtype: np.array
                    * shape: (index_num)
                asset_num_indices:
                    * dtype: list
                    * default: [5, 20, 40, 50]
                window: rolling window
                    * default: 30
        """
        self.date_num, self.asset_num = close_data.shape
        assert self.date_num == value_data.shape[0]
        assert self.asset_num == value_data.shape[1]
        self.window = window
        self.eps = eps
        self.index_num = len(indices)

        self.close_data = close_data
        self.value_data = value_data
        self.returns = returns

        self.indices = indices

        self.asset_num_indices = asset_num_indices
        self.asset_num_indices.append(self.asset_num)

        self.asset_num_dicts = {}
        self.asset_num_arr = np.arange(self.asset_num)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
            get item

            Args:
                idx: index
            Returns:
                obs_dict: observation
                    * key: asset_num
                    * value:
                        * dtype: torch.FloatTensor
                        * shape: (batch_size, asset_num, factor_num, seq_len)
                        * factor index
                            * 0: close
                            * 1: value time
                            * 2: value cs
                max_weights_dict: max return weights
                    * key: asset_num
                    * value:
                        * dtype: torch.FloatTensor
                        * shape: (batch_size, asset_num)
                rets_dict: returns
                    * key: asset_num
                    * value:
                        * dtype: torch.FloatTensor
                        * shape: (batch_size, asset_num)
        """
        index = self.indices[idx]

        obs_dict = defaultdict(torch.FloatTensor)
        max_weights_dict = defaultdict(torch.FloatTensor)
        rets_dict = defaultdict(torch.FloatTensor)

        close_data = self.close_data[index-self.window+1:index+1]
        value_data = self.value_data[index-self.window+1:index+1]

        for num in self.asset_num_indices:
            if num != self.asset_num:
                a_indices = np.random.choice(
                    self.asset_num_arr, num, replace=False)
            else:
                a_indices = self.asset_num_arr

            close_t = close_data[:, a_indices]
            value_t = value_data[:, a_indices]

            close_nm = (close_t - close_t.min(axis=0, keepdims=True)) / (
                close_t.max(axis=0, keepdims=True)
                - close_t.min(axis=0, keepdims=True) + self.eps)

            value_time_nm = (value_t - value_t.min(axis=0, keepdims=True)) / (
                value_t.max(axis=0, keepdims=True)
                - value_t.min(axis=0, keepdims=True) + self.eps)

            value_cs_nm = (value_t - value_t.min(axis=1, keepdims=True)) / (
                value_t.max(axis=1, keepdims=True)
                - value_t.min(axis=1, keepdims=True) + self.eps)

            obs = np.stack((close_nm, value_time_nm, value_cs_nm), axis=1)
            obs = obs.transpose(2, 1, 0)
            obs = torch.FloatTensor(obs.astype(float))

            # Max Return Weights
            rets = self.returns[index+1, a_indices]
            rets = (rets - rets.min(axis=-1, keepdims=True)) / (
                rets.max(axis=-1, keepdims=True)
                - rets.min(axis=-1, keepdims=True) + self.eps)
            rets = torch.FloatTensor(rets.astype(float))

            max_idx = rets.argmax()
            if len(max_idx.shape) > 0:
                max_idx = max_idx[0]
            max_weights = torch.zeros_like(rets)
            max_weights[max_idx.item()] = 1.
            max_weights = max_weights.type(torch.float)

            obs_dict[num] = obs
            max_weights_dict[num] = max_weights
            rets_dict[num] = rets

        return obs_dict, max_weights_dict, rets_dict