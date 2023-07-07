"""
    Trading BC Transformer 2 Datasets

    @author: Younghyun Kim
    Created: 2022.12.18
"""
import torch
from torch.utils.data import Dataset


class TradingBCTransformer2Dataset(Dataset):
    """
        Trading BC Transformer 2 Dataset
    """
    def __init__(self, observations, action_series, updown_series,
                asset_idx=0):
        """
            Initialization

            Args:
                observations
                    * dtype: np.array
                    * shape: (date_num, seq_len, factor_num)
                action_series: action series
                    * dtype: np.array
                    * shape: (date_num, sample_num, seq_len)
                updown_series: updown series
                    * dtype: np.array
                    * shape: (date_num, seq_len)
                asset_idx: asset index
                    * default: 0
        """
        self.date_num, self.seq_len, self.factor_num = observations.shape
        _, self.sample_num, _ = action_series.shape

        self.observations = observations
        self.action_series = action_series
        self.updown_series = updown_series

        self.asset_idx = int(asset_idx)

    def __len__(self):
        return self.date_num

    def __getitem__(self, idx):
        """
            get item

            Args:
                idx: index
            Returns:
                assets_in: asset index
                    * dtype: torch.LongTensor
                    * shape: (-1)
                obs: observations
                    * dtype: torch.FloatTensor
                    * shape: (-1, seq_len, factor_num)
                actions: action series
                    * dtype: torch.LongTensor
                    * shape: (-1, seq_len)
                updowns: updown series
                    * dtype: torch.LongTensor
                    * shape: (-1, seq_len)
        """
        assets_in = self.asset_idx
        obs = torch.FloatTensor(self.observations[idx].astype(float))

        s_idx = torch.randint(0, self.sample_num, (1,)).item()

        actions = torch.LongTensor(
            self.action_series[idx, s_idx].astype(int))

        updowns = torch.LongTensor(
            self.updown_series[idx].astype(int))

        return assets_in, obs, actions, updowns