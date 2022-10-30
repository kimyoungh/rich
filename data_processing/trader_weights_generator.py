"""
    Trader Weights Generator

    @author: Younghyun Kim
    Created: 2022.10.15
"""
import numpy as np
import pandas as pd
import cvxpy as cp


class TraderWeightsGenerator:
    """
        Trader Weights Generator
    """
    def __init__(self, returns_data=None, window=20):
        """
            Initialization

            Args:
                returns_data: returns data
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
                window: forward window
                    * default: 20
        """
        self.returns_data = returns_data
        self.window = window

    def processing_target_opt_portfolio(self, returns_data=None):
        """
            Processing Target Optimization Portfolio List

            Args:
                returns_data: returns data
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
                    * default: None
            Returns:
                weights: weights list
                    * dtype: np.array
                    * shape: (date_num-window, stock_num)
                assets: asset list
                    * dtype: np.array
                    * shape: (stock_num)
                trade_dates: trade date list
                    * dtype: np.array
                    * shape: (date_num-window)
        """
        if returns_data is None:
            returns_data = self.returns_data

        max_length = returns_data.shape[0] - self.window

        weights = []
        for t in range(max_length):
            rets = returns_data.iloc[t+1:t+self.window+1]
            w = self._calc_max_sr(rets)
            weights.append(w)
        weights = np.stack(weights, axis=0)
        assets = returns_data.columns.values

        trade_dates = returns_data.index[:max_length]

        return weights, assets, trade_dates

    def _calc_max_sr(self, returns):
        """
            Maximize Sharpe Ratio

            Args:
                returns: np.array
            Return:
                weights: np.array
                    * shape: (stock_num)
        """
        if isinstance(returns, pd.DataFrame):
            returns = returns.values

        cov = np.cov(returns.transpose())
        cov = np.nan_to_num(cov)

        mu = returns.mean(0).reshape(-1)

        weights = cp.Variable(returns.shape[1])

        objective = cp.Maximize(mu.T @ weights)

        consts = [cp.sum_squares(cov @ weights) <= 1,
                weights >= 0.]

        try:
            prob = cp.Problem(objective, consts)

            _ = prob.solve()

            weights = weights.value
            weights[weights < 0] = 0.
            weights = weights / weights.sum()
        except:
            weights = np.ones(returns.shape[1]) / returns.shape[1]

        return weights