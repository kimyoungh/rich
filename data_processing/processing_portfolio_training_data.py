"""
    Processing Module for Portfolio Training Data

    @author: Younghyun Kim
    Created on 2022.09.04
"""
import numpy as np

from data_processing.price_factor_processor import PriceFactorProcessor
from data_processing.portfolio_weights_generator import (
    PortfolioWeightsGenerator, WINDOW_LIST)

REGIME_QUANTILE = [0.01, 0.5, 0.99]


class ProcessingPortfolioTrainingData:
    """
        Processing Module for Portfolio Training Data
    """
    def __init__(self, returns_data,
                window_list=WINDOW_LIST,
                regime_target='K200', regime_window=250,
                regime_qt=REGIME_QUANTILE):
        """
            Initialization

            Args:
                returns_data: returns data
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
                window_list: forward window list
                regime_target: regime target asset code
                regime_window: regime window
        """
        self.returns_data = returns_data
        self.window_list = window_list
        self.regime_target = regime_target
        self.regime_window = regime_window
        self.regime_qt = regime_qt

        self.qt_size = len(regime_qt)

    def generate_target_data(self):
        """
            generate target data

            Returns:

        """
        factors, factors_index, trade_dates =\
            PriceFactorProcessor().calculate_mf(self.returns_data)

        returns_data = self.returns_data.loc[trade_dates]

        weights, strategies, stock_codes, trade_dates =\
            PortfolioWeightsGenerator(
                returns_data, window_list=self.window_list
                ).processing_target_portfolio()

        regimes = self.generate_regime_signal()

        factors = factors[:len(trade_dates)]
        factors_index = factors_index[:len(trade_dates)]
        regimes = regimes[:len(trade_dates)]

        return factors, weights, strategies, stock_codes,\
            trade_dates, factors_index, regimes

    def generate_opt_target_data(self):
        """
            generate opt target data
        """
        factors, factors_index, trade_dates =\
            PriceFactorProcessor().calculate_mf(self.returns_data)

        returns_data = self.returns_data.loc[trade_dates]

        weights, strategies, stock_codes, trade_dates =\
            PortfolioWeightsGenerator(
                returns_data, window_list=self.window_list
                ).processing_target_opt_portfolio()

        regimes = self.generate_regime_signal()

        factors = factors[:len(trade_dates)]
        factors_index = factors_index[:len(trade_dates)]
        regimes = regimes[:len(trade_dates)]

        return factors, weights, strategies, stock_codes,\
            trade_dates, factors_index, regimes

    def generate_regime_signal(self):
        """
            Generate Regime Signal
        """
        rets_reg = self.returns_data[self.regime_target]

        mu = rets_reg.iloc[::-1].rolling(
            self.regime_window,
            min_periods=2).mean().iloc[::-1].shift(-1)
        sig = rets_reg.iloc[::-1].rolling(
            self.regime_window,
            min_periods=2).std().iloc[::-1].shift(-1)

        sr = mu / sig
        sr = sr.ffill()

        regimes = np.zeros_like(sr)

        for i, qt in enumerate(self.regime_qt):
            if i == 0:
                regimes[sr < sr.quantile(qt)] = i
            elif i < (self.qt_size - 1):
                regimes[(sr >= sr.quantile(qt_prev)) &
                        (sr < sr.quantile(qt))] = i
            else:
                regimes[sr >= sr.quantile(qt)] = i

            qt_prev = qt
        regimes = regimes.astype(int)

        return regimes