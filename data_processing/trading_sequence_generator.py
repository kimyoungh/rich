"""
    Data Processor for Trading

    @author: Younghyun Kim
    Created on 2022.11.20
"""
from copy import deepcopy
from collections import defaultdict
import numpy as np
import pandas as pd


class TradingSequenceGenerator:
    """
        Decision Transformer를 위한
        Investment Strategy Sequence 생성
        * Simulated Annealing 기반
        * Action: [Sell, Buy]
    """
    def __init__(self, price, window=24, fee=0.001,
                reward_list=[-2, 0, +1],
                trading_period=24,
                sample_num=5,
                decay=0.99, eps=1e-6):
        """
            Args:
                price: price series
                    * dtype: pd.DataFrame
                    * shape: (date_num, 6)
                window: rolling window for observations
                    * default: 24
                fee: trading fee
                    * default: 0.1%
                reward_list: reward list(worst, neutral, best)
                    * dtype: list
                    * default: [-2, 0, +1]
                trading_period: trading period
                    * default: 1440 minutes(1-day)
                sample_num: sample number
                    * default: 5
                decay: temperature reduction coefficient
                eps: epsilon
        """
        self.price = price[['open', 'high', 'low', 'close', 'value']]
        self.window = window
        self.returns = price[['close']].iloc[window-1:].pct_change()
        self.returns.iloc[0] = 0.
        self.date_num = self.returns.shape[0]

        self.action_num = 2  # [Sell, Buy]
        self.actions = np.arange(self.action_num)

        self.fee = fee
        self.reward_list = np.array(reward_list)
        self.trading_period = trading_period
        self.sample_num = sample_num

        self.decay = decay
        self.eps = eps

        if self.date_num < self.trading_period:
            raise Exception("Too short returns series!")

        self.date_length = self.date_num - self.trading_period
        self.max_value = reward_list[2] * trading_period
        self.min_value = reward_list[0] * trading_period

        self.value_table = np.arange(
            self.min_value, self.max_value+ 1)

    def generate_overall_dataset(self, pick_num=1, sample_num=None,
                                temp=20000, eps=None, rn_calc=False):
        """
            generate overall strategy series dataset
            Args:
                pick_num: picking number of strategies for neighbor_op
                sample_num: sample number for each date
                    * default: self.sample_num
                temp: temparature
                eps: epsilon
            Returns:
                dataset: dict of dataset
                    * key: data name
                    * value:
                        observations: observations
                            * dtype: np.array
                            * shape: (date_num-trading_period, trading_period, ohlcv_num)
                        action_series: action series
                            * dtype: np.array
                            * shape: (date_num-trading_period, sample_num, trading_period)
                        rets_series: return series
                            * dtype: np.array
                            * shape: (date_num-trading_period, sample_num, trading_period)
                        rew_series: rewards series
                            * dtype: np.array
                            * shape: (date_num-trading_period, sample_num, trading_period)
                        val_series: value series
                            * dtype: np.array
                            * shape: (date_num-trading_period, sample_num, trading_period)
                        date_series: date series
                            * dtype: np.array
                            * shape: (date_num-trading_period)
        """
        # Best Series
        dataset_best = self.calculate_strategy_series_dataset(
            mode='best', pick_num=pick_num, sample_num=sample_num,
            temp=temp, eps=eps)
        print("best done!")

        if rn_calc:
            # Worst Series
            dataset_worst = self.calculate_strategy_series_dataset(
                mode='worst', pick_num=pick_num, sample_num=sample_num,
                temp=temp, eps=eps)
            print("worst done!")

            # Random Series
            dataset_random = self.calculate_strategy_series_dataset(
                mode='random', pick_num=pick_num, sample_num=sample_num,
                temp=temp, eps=eps)
            print("random done!")

        pmax = self.price.rolling(self.window).max().dropna()
        pmin = self.price.rolling(self.window).min().dropna()
        pnorm = (self.price.iloc[self.window-1:] - pmin) / (pmax - pmin)

        obs = pnorm.values

        observations = []
        for i in range(obs.shape[0] - self.trading_period):
            observations.append(obs[i:i+self.trading_period])

        dataset = defaultdict(np.array)

        if rn_calc:
            dataset['action_series'] = np.concatenate(
                (dataset_best['action_series'], dataset_random['action_series'],
                dataset_worst['action_series']), axis=1)
            dataset['rets_series'] = np.concatenate(
                (dataset_best['rets_series'], dataset_random['rets_series'],
                dataset_worst['rets_series']), axis=1)
            dataset['rew_series'] = np.concatenate(
                (dataset_best['rew_series'], dataset_random['rew_series'],
                dataset_worst['rew_series']), axis=1)
            dataset['val_series'] = np.concatenate(
                (dataset_best['val_series'], dataset_random['val_series'],
                dataset_worst['val_series']), axis=1)
        else:
            dataset = dataset_best
        dataset['observations'] = np.stack(observations, axis=0)
        dataset['date_series'] = dataset_best['date_series']

        return dataset

    def calculate_strategy_series_dataset(self, mode='best',
                                        pick_num=1, sample_num=None,
                                        temp=20000, eps=None):
        """
            calculate series of strategy series

            Args:
                mode: Style of strategy series
                    * kind: [best, random, worst]
                    * default: best
                pick_num: picking number of strategies for neighbor_op
                sample_num: sample number for each date
                    * default: self.sample_num
                temp: temparature
                eps: epsilon
            Returns:
                dataset: dict of dataset
                    * key: data name
                    * value:
                        action_series: action series
                            * dtype: np.array
                            * shape: (date_num-trading_period, sample_num, trading_period)
                        rets_series: return series
                            * dtype: np.array
                            * shape: (date_num-trading_period, sample_num, trading_period)
                        rew_series: rewards series
                            * dtype: np.array
                            * shape: (date_num-trading_period, sample_num, trading_period)
                        val_series: value series
                            * dtype: np.array
                            * shape: (date_num-trading_period, sample_num, trading_period)
                        date_series: date series
                            * dtype: np.array
                            * shape: (date_num-trading_period)
        """
        if sample_num is None:
            sample_num = self.sample_num

        action_series, rets_series,\
            rew_series, val_series = [], [], [], []

        for i in range(self.date_length):
            actions, rets, rews, vals = [], [], [], []
            for _ in range(sample_num):
                if mode != 'random':
                    action_top, ret_top, rew_top, val_top =\
                        self.search_strategy_series(
                            self.returns.iloc[
                                i+1:i+self.trading_period+1].values,
                            mode=mode, pick_num=pick_num, temp=temp,
                            eps=eps)
                else:
                    action_top, ret_top, rew_top, val_top =\
                        self.random_strategy_series(
                            self.returns.iloc[
                                i+1:i+self.trading_period+1].values)

                actions.append(action_top)
                rets.append(ret_top)
                rews.append(rew_top)
                vals.append(val_top)

            actions = np.stack(actions, axis=0)
            rets = np.stack(rets, axis=0)
            rews = np.stack(rews, axis=0)
            vals = np.stack(vals, axis=0)

            action_series.append(actions)
            rets_series.append(rets)
            rew_series.append(rews)
            val_series.append(vals)

        action_series = np.stack(action_series, axis=0)
        rets_series = np.stack(rets_series, axis=0)
        rew_series = np.stack(rew_series, axis=0)
        val_series = np.stack(val_series, axis=0)

        date_series = self.returns.index.values[:self.date_length]

        dataset = defaultdict(np.array)
        dataset['action_series'] = action_series
        dataset['rets_series'] = rets_series
        dataset['rew_series'] = rew_series
        dataset['val_series'] = val_series
        dataset['date_series'] = date_series

        return dataset

    def search_strategy_series(self, rets, mode='best', pick_num=1,
                            temp=20000, eps=None):
        """
            Search method for picking strategy series based on mode,
            by Simulated Annealing

            Args:
                rets: returns series
                    * dtype: np.array
                    * shape: (trading_period, strategy_num)
                mode: Style of strategy series
                    * kind: [best, worst]
                    * default: best
                pick_num: picking number of strategies for neighbor_op
                temp: temparature
                eps: epsilon
            Returns:
                series_top: best strategy series
                    * dtype: np.array
                    * shape: (trading_period)
                returns_top: return series of best strategy series
                    * dtype: np.array
                    * shape: (trading_period)
                rewards_top: reward series of best strategy series
                    * dtype: np.array
                    * shape: (trading_period)
                values_top: value series of best strategy series
                    * dtype: np.array
                    * shape: (trading_period)
        """
        if eps is None:
            eps = self.eps

        series_now = self.pick_initial_strategy_series()

        rs, rewards, values = self.calculate_series_values(
            series_now, rets)

        values_now = values

        series_top = series_now
        returns_top = rs
        rewards_top = rewards
        values_top = values

        while temp >= eps:
            series_new = self.neighbor_op(series_now, pick_num)

            returns_new, rewards_new, values_new =\
                self.calculate_series_values(series_new, rets)

            if mode == 'best':
                cost = values_new[0] - values_now[0]
                cost_top = values_new[0] - values_top[0]
            elif mode == 'worst':
                cost = values_now[0] - values_new[0]
                cost_top = values_top[0] - values_new[0]

            if cost > 0:
                series_now = series_new
                values_now = values_new
            else:
                temp_v = cost / temp
                if np.random.uniform(0, 1) < np.exp(temp_v):
                    series_now = series_new
                    values_now = values_new

            if cost_top > 0:
                series_top = series_new
                returns_top = returns_new
                rewards_top = rewards_new
                values_top = values_new

            temp = temp * self.decay

        return series_top, returns_top, rewards_top, values_top

    def random_strategy_series(self, rets):
        """
            Get Random Strategy Series

            Args:
                rets: returns series
                    * dtype: np.array
                    * shape: (trading_period, strategy_num)
            Returns:
                series_top: best strategy series
                    * dtype: np.array
                    * shape: (trading_period)
                returns_top: return series of best strategy series
                    * dtype: np.array
                    * shape: (trading_period)
                rewards_top: reward series of best strategy series
                    * dtype: np.array
                    * shape: (trading_period)
                values_top: value series of best strategy series
                    * dtype: np.array
                    * shape: (trading_period)
        """
        series_now = self.pick_initial_strategy_series()

        returns_now, rewards_now, values_now =\
            self.calculate_series_values(series_now, rets)

        return series_now, returns_now, rewards_now, values_now

    def neighbor_op(self, picked, pick_num=1):
        """
            pick neighbor of picked strategy series

            Args:
                picked: previous picked strategy series
                    * dtype: np.array
                    * shape: (trading_period)
                pick_num: picking number of strategies for neighbor_op
                    * default: 1
        """
        picked_new = deepcopy(picked)
        pick_num = min(pick_num, self.trading_period)

        prng = np.arange(len(picked_new))
        prng = np.random.choice(prng, pick_num, replace=False)

        for p in prng:
            action_p = np.random.choice(self.actions, 1).item()
            picked_new[p] = action_p

        return picked_new

    def pick_initial_strategy_series(self):
        """
            Pick Initial Strategy Series
        """
        picked = np.random.choice(self.actions,
                                self.trading_period,
                                replace=True)

        return picked

    def calculate_series_values(self, series, returns):
        """
            Calculate Series Values
                * return series, reward series, value series

            Args:
                series: picked strategy series
                    * dtype: np.array
                    * shape: (trading_period)
                returns: return series
                    * dtype: np.array
                    * shape: (trading_period, 1)
            Return:
                rets: strategy return series
                    * dtype: np.array
                    * shape: (trading_period)
                rewards: strategy reward series
                    * dtype: np.array
                    * shape: (trading_period)
                values: strategy value series
                    * dtype: np.array
                    * shape: (trading_period)
        """
        rets = []
        for i, action in enumerate(series):
            ret_t = returns[i, 0]
            if action == 0:
                if ret_t < 0:
                    ret = -ret_t
                else:
                    ret = 0.
            else:
                ret = ret_t

            if i == 0:
                rets.append(ret)
            else:
                if action != action_prev:
                    ret = ret - (self.fee * 2)

                rets.append(ret)

            action_prev = action

        rets = np.array(rets)
        rewards = np.where(rets > 0., self.reward_list[2],
                        np.where(rets == 0, self.reward_list[1],
                                self.reward_list[0]))

        values = rewards[::-1].cumsum()[::-1]

        return rets, rewards, values