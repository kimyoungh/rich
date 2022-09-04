"""
    Processing Module for Portfolio Training Data

    @author: Younghyun Kim
    Created on 2022.09.04
"""
from data_processing.price_factor_processor import PriceFactorProcessor
from data_processing.portfolio_weights_generator import (
    PortfolioWeightsGenerator, WINDOW_LIST)


class ProcessingPortfolioTrainingData:
    """
        Processing Module for Portfolio Training Data
    """
    def __init__(self, returns_data,
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

        factors = factors[:len(trade_dates)]
        factors_index = factors_index[:len(trade_dates)]

        return factors, weights, strategies, stock_codes,\
            trade_dates, factors_index