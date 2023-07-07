"""
    Data Processor for Trading GPT

    @author: Younghyun Kim
    Created on 2023.03.19
"""
from collections import defaultdict
import numpy as np
import pandas as pd


class TradingGPTDataGenerator:
    """
        TradingGPT를 위한 data 생성
    """
    def __init__(self, price, window=250,
                eps=1e-6):
        """
            Args:
                price: price series
                    * dtype: pd.DataFrame
                    * shape: (date_num, 6)
                window: rolling window for observations
                    * default: 250
        """
        self.price = price[['open', 'high', 'low', 'close', 'value']]
        self.window = window
        self.returns = price[['close']].iloc[window-1:].pct_change()
        self.returns.iloc[0] = 0.

        self.date_num = len(self.returns)

        self.eps = eps

    def generate_dataset(self):
        """
            generate overall dataset

            Return:
                dataset: dict of dataset
                    * key: data name
                    * value:
                        observations:
                            * dtype: np.array
                            * shape: (date_num-1, factor_num)
                        returns:
                            * dtype: np.array
                            * shape: (date_num-1, 1)
        """
        ohlc = self.price[
            ['open', 'high', 'low', 'close']].iloc[self.window-1:].copy()
        o_values = ohlc.values
        o_max = o_values.max(axis=1, keepdims=True)
        o_min = o_values.min(axis=1, keepdims=True)

        o_values = (o_values - o_min) / (o_max - o_min + self.eps)
        ohlc = pd.DataFrame(o_values, columns=ohlc.columns,
                        index=ohlc.index)[['open', 'close']]

        pmax = self.price.rolling(self.window).max().dropna()
        pmin = self.price.rolling(self.window).min().dropna()
        pnorm = (self.price.iloc[self.window-1:] - pmin) / (
            pmax - pmin + self.eps)
        direc = (self.returns > 0).astype(int)

        pnorm = pd.concat((ohlc, pnorm, direc), axis=1)

        obs = pnorm.values[:-1]
        returns = self.returns.iloc[1:].values.ravel()

        dataset = defaultdict(np.array)

        dataset['observations'] = obs
        dataset['returns'] = returns
        dataset['timestamps'] = self.returns.index[1:].values

        return dataset