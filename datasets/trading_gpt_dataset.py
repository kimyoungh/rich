"""
    Trading BC Transformer 2 Datasets

    @author: Younghyun Kim
    Created: 2022.12.18
"""
from collections import defaultdict
import torch
from torch.utils.data import Dataset


class TradingGPTDataset(Dataset):
    """
        Trading GPT Dataset
    """
    def __init__(self, datasets, indices,
                trading_period=60):
        """
            Initialization

            Args:
                datasets
                    * dtype: dict
                    * keys: ['observations', 'returns']
                    * value:
                        observations:
                            * dtype: np.array
                            * shape: (date_num, factor_num)
                        returns:
                            * dtype: np.array
                            * shape: (date_num, 1)
                indices
                    * dtype: np.array
                    * shape: (data_length)
                trading_period: trading period
                    * default: 60
        """
        self.observations = datasets['observations']
        self.returns = datasets['returns']
        self.trading_period = trading_period

        self.date_num, self.factor_num = self.observations.shape
        self.indices = indices

        assert indices[-1] <= (self.date_num - trading_period)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
            get item

            Args:
                idx: index
            Returns:
                obs: observations
                    * dtype: torch.FloatTensor
                    * shape: (-1, seq_len, factor_num)
                actions: action series
                    * dtype: torch.LongTensor
                    * shape: (-1, seq_len)
                returns: return series
                    * dtype: torch.LongTensor
                    * shape: (-1, seq_len)
        """
        t_idx = self.indices[idx]

        # Observations
        obs = torch.FloatTensor(
            self.observations[
                t_idx:t_idx+self.trading_period].astype(float))

        # Returns
        returns = torch.FloatTensor(
            self.returns[t_idx:t_idx+self.trading_period].astype(float))
        returns = returns.view(-1)

        # Actions(Updown Prediction)
        ## 0: Up(Buy)
        ## 1: Down(Sell)
        actions = (returns <= 0).type(torch.long)

        return obs, actions, returns