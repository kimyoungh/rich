"""
    Processing Module for Portfolio Training Data

    @author: Younghyun Kim
    Created on 2022.09.04
"""
import numpy as np

from data_processing.price_factor_processor import PriceFactorProcessor
from data_processing.trader_weights_generator import TraderWeightsGenerator


class ProcessingTraderTrainingData:
    """
        Processing Module for Portfolio Training Data
    """
    def __init__(self, returns_data, greturns_data, window=20):
        """
            Initialization

            Args:
                returns_data: returns data
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
                greturns_data: greturns data
                    * dtype: pd.DataFrame
                    * shape: (date_num, asset_num)
                window_list: forward window
                    * default: 20
        """
        self.returns_data = returns_data
        self.greturns_data = greturns_data
        self.window = window

    def generate_target_data(self):
        """
            generate opt target data
        """
        factors, factors_index, trade_dates =\
            PriceFactorProcessor().calculate_mf(self.returns_data)

        gfactors, gfactors_index, gtrade_dates =\
            PriceFactorProcessor().calculate_mf(self.greturns_data)

        returns_data = self.returns_data.loc[trade_dates]

        weights, stock_codes, trade_dates =\
            TraderWeightsGenerator(returns_data,
                self.window).processing_target_opt_portfolio()

        factors = factors[:len(trade_dates)]
        factors_index = factors_index[:len(trade_dates)]
        gfactors = gfactors[:len(trade_dates)]
        gfactors_index = gfactors_index[:len(trade_dates)]

        return_series = []
        for i in range(len(returns_data) - self.window):
            rets = returns_data.values[i+1:i+self.window+1]
            return_series.append(rets)

        return_series = np.stack(return_series, axis=0)

        return factors, gfactors, weights, stock_codes,\
            trade_dates, gtrade_dates,\
                factors_index, gfactors_index, return_series