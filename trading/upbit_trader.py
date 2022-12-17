"""
    Upbit Trader

    @author: Younghyun Kim
    Created on 2022.12.11
"""
import os
import time
import datetime
import numpy as np
import pandas as pd
import torch

import pyupbit

from models.cfg.trading_bc_transformer_config import TRADING_BC_TRANSFORMER_CONFIG
from models.trading_bc_transformer import TradingBCTransformer
from trading.upbit_trader_config import UPBIT_TRADER_CONFIG


class UpbitTrader:
    """
        Upbit Trader
    """
    def __init__(self, config=None):
        """
            Initialization
        """
        if config is None:
            config = UPBIT_TRADER_CONFIG
        self.config = config
        self.asset = config['asset']
        self.window = config['window']
        self.interval = config['interval']
        self.asset_idx = config['asset_idx']
        self.trading_data_path = config['trading_data_path']
        self.upbit_access_token = config['upbit_access_token']
        self.upbit_secret_key = config['upbit_secret_key']

        self.device = config['device']

        self.upbit = pyupbit.Upbit(self.upbit_access_token,
                                self.upbit_secret_key)

        chance = self.upbit.get_chance(self.asset)
        self.bid_fee = float(chance['bid_fee'])
        self.ask_fee = float(chance['ask_fee'])

        if not os.path.exists(self.trading_data_path):
            os.mkdir(self.trading_data_path)

        self.model = TradingBCTransformer(TRADING_BC_TRANSFORMER_CONFIG)
        self.model.eval()
        self.model.load_state_dict(
            torch.load(config['model_path'],
            map_location=self.device))

        self.assets_in = torch.LongTensor([self.asset_idx]).view(-1)

    def execute_trading(self):
        """
            Execute Trading
        """
        balance_prev = self.get_total_balance()
        obs = torch.FloatTensor().to(self.device)

        if self.upbit.get_balance(self.asset) > 0:
            action = 1
        else:
            action = 0

        cnt = 0

        print(balance_prev)

        dec = True
        price_prev = None

        while True:
            now = datetime.datetime.now()

            while dec and now.minute <= 2:
                print(now)
                print('make decision..')
                obs_t, price = self.get_observation()

                if (price_prev is not None and\
                    price.index[-1] != price_prev.index[-1]) or\
                    (price_prev is None):

                    pret = (
                        price['close'].iloc[-1] / price['close'].iloc[-2]
                        ) - 1.

                    balance = self.get_total_balance()

                    ret = (balance / balance_prev) - 1.

                    if action == 1:
                        if ret > 0:
                            reward = +1.
                        elif ret < 0:
                            reward = -2.
                        else:
                            reward = 0.
                    elif action == 0 and pret < 0:
                        reward = +1.
                    else:
                        reward = 0.

                    obs = torch.cat((obs, obs_t), dim=1)

                    if obs.shape[1] > self.window:
                        obs = obs[:, -self.window:]

                    if cnt == 0:
                        actions_in = None
                    elif cnt > 0:
                        actions_in = action_preds

                    if cnt == 0:
                        rewards_in = None
                    elif cnt == 1:
                        rewards_in = torch.FloatTensor(
                            [reward]).view(1, -1).to(self.device)
                    else:
                        rewards_in = torch.cat(
                            (rewards_in,
                            torch.FloatTensor(
                                [reward]).view(1, -1).to(self.device)),
                            dim=1)

                        if rewards_in.shape[1] > (self.window-1):
                            rewards_in = rewards_in[:, -(self.window-1):]

                    action_preds, _ = self.model.inference(
                        self.assets_in, obs, actions_in, rewards_in)

                    action_next = int(action_preds[0, -1].item())

                    if action_next == 1 and action == 0:
                        balance_t = balance * (1. - self.bid_fee)
                        rst = self.upbit.buy_market_order(
                            self.asset, balance_t)
                        print(rst)
                        print('buy: ', balance_t)
                    elif action_next == 0 and action == 1:
                        balance_t = self.upbit.get_balance(self.asset)
                        rst = self.upbit.sell_market_order(
                            self.asset, balance_t)
                        print(rst)
                        print('sell: ', balance_t)
                    print(action_next)

                    action = action_next
                    balance_prev = balance
                    price_prev = price
                    cnt += 1
                    dec = False
                    time.sleep(60 * 3)
                else:
                    dec = True
            dec = True

    def get_total_balance(self):
        """
            get total balance
        """
        balance_dict = self.upbit.get_balances()
        balance = 0
        for dicts in balance_dict:
            if dicts['currency'] == 'KRW':
                balance += float(dicts['balance'])
            else:
                curr = dicts['currency']
                c_price = pyupbit.get_current_price(curr)
                balance += float(dicts['balance']) * c_price

        return balance

    def get_observation(self):
        """
            get observation

            Return:
                obs_v: observations
                    * dtype: torch.FloatTensor
                    * shape: (1, 1, factor_num)
                price: price series
                    * dtype: pd.DataFrame
                    * shape: (window, 5)
        """
        price = pyupbit.get_ohlcv(self.asset, interval=self.interval,
                                count=self.window)
        price = price[['open', 'high', 'low', 'close', 'value']]

        pmax = price.rolling(self.window).max().dropna()
        pmin = price.rolling(self.window).min().dropna()
        obs = (price.iloc[self.window-1:] - pmin) / (pmax - pmin)

        obs_v = obs.values
        obs_v = torch.FloatTensor(obs_v.astype(float)).view(1, 1, -1)
        obs_v = obs_v.to(self.device)

        return obs_v, price


if __name__ == "__main__":
    trader = UpbitTrader(UPBIT_TRADER_CONFIG)
    trader.execute_trading()