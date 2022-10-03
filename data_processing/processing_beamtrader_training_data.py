"""
    Processing Module for BeamTrader Training Data

    @author: Younghyun Kim
    Created on 2022.10.02
"""
import numpy as np

from data_processing.price_factor_processor import PriceFactorProcessor
from data_processing.trading_seq_generator import TradingSeqGenerator

REGIME_QUANTILE = [0.01, 0.5, 0.99]


class ProcessingBeamTraderTrainingData:
    """
        Processing Module for BeamTrader Training Data
    """
    def __init__(self, returns_data, seq_len=20,
                trading_fee=0.01,
                beam_width=10, n_expand=20, discount=0.99,
                regime_target='K200', regime_window=20,
                regime_qt=REGIME_QUANTILE, device='cpu'):
        """
            Initialization

            Args:
                returns_data: returns data
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
                seq_len: forward sequence length
                trading_fee: trading fee
                    * default: 0.01
                beam_width: beam width
                    * default: 10
                n_expand: number of expansion
                    * default: 20
                discount: discount
                    * default: 0.99
                regime_target: regime target asset code
                regime_window: regime window
        """
        self.returns_data = returns_data
        self.seq_len = seq_len
        self.trading_fee = trading_fee
        self.beam_width = beam_width
        self.n_expand = n_expand
        self.discount = discount
        self.device = device
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

        best_seqs, best_rews, best_vals, indices =\
            TradingSeqGenerator(returns_data, self.seq_len,
                    self.trading_fee, self.beam_width,
                    self.n_expand, self.discount,
                    self.device).generate_beam_sequences(rebal=False)

        best_rebal_seqs, best_rebal_rews, best_rebal_vals, _ =\
            TradingSeqGenerator(returns_data, self.seq_len,
                    self.trading_fee, self.beam_width,
                    self.n_expand, self.discount,
                    self.device).generate_beam_sequences(rebal=True)

        regimes = self.generate_regime_signal()

        factors = factors[:len(indices)]
        factors_index = factors_index[:len(indices)]
        regimes = regimes[:len(indices)]

        return factors, best_seqs, best_rebal_seqs,\
            best_rews, best_rebal_rews,\
            best_vals, best_rebal_vals,\
            trade_dates[indices], factors_index, regimes

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