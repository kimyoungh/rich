"""
    Portfolio Weights Generator

    @author: Younghyun Kim
    Created: 2022.09.04
"""
import numpy as np
import pandas as pd

WINDOW_LIST = [5, 20, 60, 120, 250]


class PortfolioWeightsGenerator:
    """
        Portfolio Weights Generator
    """
    def __init__(self, returns_data=None,
                window_list=WINDOW_LIST):
        """
            Initialization

            Args:
                returns_data: returns data
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
                window_list: forward window list
        """
        self.returns_data = returns_data
        self.window_list = window_list

        self.strategies = {
            'fwd_price_momentum': 'desc',
            'fwd_rsi': 'desc',
            'fwd_skew': 'asc',
            'fwd_lpm': 'asc',
        }

    def processing_target_portfolio(self, returns_data=None):
        """
            Processing Target Portfolio List

            Args:
                returns_data: returns data
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
                    * default: None
            Returns:
                weights_list: weights list
                    * dtype: np.array
                    * shape: (strategy_num, date_num-window, stock_num)
                strategy_list: strategy list
                    * dtype: list
                    * shape: (strategy_num)
                assets: asset list
                    * dtype: np.array
                    * shape: (stock_num)
                trade_dates: trade date list
                    * dtype: np.array
                    * shape: (date_num-window)
        """
        if returns_data is None:
            returns_data = self.returns_data

        max_length = returns_data.shape[0] - max(self.window_list)

        weights_list, strategy_list = [], []
        for strategy, align in self.strategies.items():
            for window in self.window_list:
                values = getattr(self, "_calc_"+strategy)(
                    returns_data, window)

                if align == 'desc':
                    onehot = pd.get_dummies(
                        values.values.argmax(axis=1))
                elif align == 'asc':
                    onehot = pd.get_dummies(
                        values.values.argmin(axis=1))

                weights = onehot.astype(float)[:max_length]
                weights_list.append(weights)

                strategy_list.append(strategy+"_"+str(window))

        weights_list = np.stack(weights_list, axis=0)
        assets = returns_data.columns.values

        trade_dates = returns_data.index[:max_length]

        return weights_list, strategy_list, assets, trade_dates

    def _calc_fwd_price_momentum(self, returns, window=5):
        """
            calculate forward price returns

            Args:
                returns: returns data
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
                window: target window
            Return:
                pmom: forward price returns
                    * dtype: pd.DataFrame
                    * shape: (date_num-window, stock_num)
        """
        rets = np.log(returns + 1.)
        rets = rets.iloc[::-1]

        pmom = rets.rolling(window).sum().shift(1)
        pmom = pmom.iloc[::-1].dropna()
        pmom = np.exp(pmom) - 1.

        return pmom

    def _calc_fwd_rsi(self, returns, window=5):
        """
            calculate forward rsi

            Args:
                returns: returns data
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
                window: target window
            Return:
                rsi: forward rsi
                    * dtype: pd.DataFrame
                    * shape: (date_num-window, stock_num)
        """
        rets = returns.iloc[::-1]
        au = (rets > 0).astype(float).rolling(
            window, min_periods=window).mean()
        ad = (rets <= 0).astype(float).rolling(
            window, min_periods=window).mean()

        rsi = au / (au + ad)
        rsi = rsi.shift(1).iloc[::-1].dropna()

        return rsi

    def _calc_fwd_skew(self, returns, window=5):
        """
            calculate forward skew

            Args:
                returns: returns data
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
                window: target window
            Return:
                skew: forward skew
                    * dtype: pd.DataFrame
                    * shape: (date_num-window, stock_num)
        """
        rets = returns.iloc[::-1]
        skew = rets.rolling(window, min_periods=window).skew()

        skew = skew.shift(1).iloc[::-1].dropna()

        return skew

    def _calc_fwd_lpm(self, returns, window=5):
        """
            calculate forward lower partial moment

            Args:
                returns: returns data
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
                window: target window
            Return:
                lpm: forward lpm
                    * dtype: pd.DataFrame
                    * shape: (date_num-window, stock_num)
        """
        lpm = 0. - returns
        lpm = lpm.iloc[::-1]

        lpm = lpm.where(lpm > 0, 0).rolling(
            window, min_periods=window).mean()

        lpm = lpm.shift(1).iloc[::-1].dropna()

        return lpm