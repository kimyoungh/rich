"""
    Price Factor Processor for Trading

    @author: Younghyun Kim
    Created: 2022.09.03
"""
import numpy as np
import pandas as pd


class PriceFactorProcessor:
    """
        Data Processor for Stocks Price Factors
    """
    def __init__(self, returns_data: pd.DataFrame = None,
                eps: float = 1e-6):
        """
            Initialization

            Args:
                returns_data: returns data
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
                eps: epsilon
        """
        self.returns_data = returns_data

        self.window_list = [5, 10, 20, 40, 60, 90, 120, 250]

        self.price_points = np.array([-250, -120, -90, -60,
                                    -40, -20, -10, -5, -1])
        self.eps = eps

    def calculate_mf(self, returns_data: pd.DataFrame = None,
                    normalize=True):
        """
            calculate mf

            Args:
                returns_data: returns data
                    * dtype: pd.DataFrame
                    * shape: ( date_num, stock_num)
        """
        if returns_data is None:
            returns_data = self.returns_data

        factors = pd.DataFrame()
        for w in self.window_list:
            pm = self._calc_price_momentum(returns_data, window=w)
            pm = pm.stack(dropna=False)
            factors = pd.concat((
                factors, pm.to_frame('pm_'+str(w))), axis=1)

            rsi = self._calc_rsi(returns_data, window=w)
            rsi = rsi.stack(dropna=False)
            factors = pd.concat((
                factors, rsi.to_frame('rsi_'+str(w))), axis=1)

            vol = self._calc_volatility(returns_data, window=w)
            vol = vol.stack(dropna=False)
            factors = pd.concat((
                factors, vol.to_frame('vol_'+str(w))), axis=1)

            lpm = self._calc_lpm(returns_data, window=w)
            lpm = lpm.stack(dropna=False)
            factors = pd.concat((
                factors, lpm.to_frame('lpm_'+str(w))), axis=1)

        factors = factors.reset_index()

        returns_index = returns_data.index

        factors = factors.rename(
            columns={'level_0': 'trade_date', 'level_1': 'code'})

        factors = factors[
            factors['trade_date'] >= returns_index[
                self.window_list[-1] - 1]]
        factors = factors.where(pd.notnull(factors), None)

        if normalize:
            factors = self.minmax_scaling(factors)

        factors_v = factors.values[:, 2:].reshape(
            -1, returns_data.shape[1], factors.shape[-1] - 2)
        factors_index = factors.values[:, :2].reshape(
            -1, returns_data.shape[1], 2)

        # Calculate Price Point Data
        price_series = (1 + returns_data).cumprod()
        price_data, _ = self._calc_price_point(
            price_series, normalize=normalize)

        assert factors_v.shape[:2] == price_data.shape[:2]

        factors_v = np.concatenate(
            (factors_v, price_data), axis=-1)

        trade_dates = factors_index[:, 0, 0]

        return factors_v, factors_index, trade_dates

    def minmax_scaling(self, factors):
        """
            minmax scaling
        """
        dates = factors['trade_date'].unique()
        codes = factors['code'].unique()

        factors_v = factors.values
        factors_index_v = factors.values[:, :2]
        factors_v = factors_v[:, 2:].reshape(
            dates.shape[0], codes.shape[0], -1)

        fmax = factors_v.max(1, keepdims=True)
        fmin = factors_v.min(1, keepdims=True)

        normalized = (factors_v - fmin) / (fmax - fmin + self.eps)

        normalized = normalized.reshape(-1, normalized.shape[-1])

        normalized = np.concatenate(
            (factors_index_v, normalized), axis=1)

        normalized = pd.DataFrame(normalized, columns=factors.columns)
        normalized = normalized.where(pd.notnull(normalized), -1)

        return normalized

    def _calc_price_point(self, price_series, normalize=True):
        """
            calculate price point series
        """
        max_window = max(abs(self.price_points))
        length, stock_num = price_series.shape

        price_data = []
        price_index_data = []

        for i in range(length - max_window + 1):
            pdata = price_series.iloc[
                i:i+max_window].transpose().iloc[:, self.price_points]
            price_data.append(pdata.values)
            price_index_data.append(
                list(zip(
                    [price_series.index[i+max_window-1]] * stock_num,
                    pdata.index.values)))

        price_data = np.stack(price_data, axis=0)
        price_index_data = np.stack(price_index_data, axis=0)

        if normalize:
            pmax = price_data.max(-1, keepdims=True)
            pmin = price_data.min(-1, keepdims=True)

            price_data = (price_data - pmin) / (pmax - pmin + self.eps)

        return price_data, price_index_data

    def _calc_price_momentum(self, returns, window=5, log=False):
        """
            calculate price returns

            Args:
                returns: returns data
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
                window: moving window
                log: 로그수익률 여부
            Return:
                pmom: price momentums
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
        """
        if not log:
            rets = np.log(returns + 1.)
        else:
            rets = returns.copy()

        pmom = rets.rolling(window, min_periods=window).sum()
        pmom = np.exp(pmom) - 1.

        return pmom

    def _calc_rsi(self, returns, window=5):
        """
            calculate relative strength indicators

            Args:
                returns: returns data
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
                window: moving window
            Return:
                rsi: rsi
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
        """
        au = (returns > 0).astype(float).rolling(
            window, min_periods=window).mean()
        ad = (returns <= 0).astype(float).rolling(
            window, min_periods=window).mean()

        rsi = au / (au + ad)

        return rsi

    def _calc_volatility(self, returns, window=5):
        """
            calculate volatility

            Args:
                returns: returns data
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
                window: moving window
            Return:
                volatility
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
        """
        volatility = returns.rolling(window, min_periods=window).std()

        return volatility

    def _calc_skew(self, returns, window=5):
        """
            calculate skewness
            Args:
                returns: returns data
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
                window: moving window
            Return:
                skew
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
        """
        skew = returns.rolling(window, min_periods=window).skew()

        return skew

    def _calc_lpm(self, returns, window=5, tau=0.):
        """
            calculate lower partial moment

            Args:
                returns: returns data
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
                window: moving window
                tau: return level criterion
                    * default: 0.
            Return:
                lpm
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
        """
        lpm = tau - returns
        lpm = lpm.where(lpm > 0, 0).rolling(
            window, min_periods=window).mean()

        return lpm

    def _calc_p_up(self, rolling_returns):
        """
            calculate probability of up-returns of all stocks

            Args:
                rolling_returns: pd.DataFrame(date_num, 1)
        """
        ups = (rolling_returns > 0.).astype(float).mean(1)
        ups = pd.DataFrame(ups, columns=['p_up'])

        return ups