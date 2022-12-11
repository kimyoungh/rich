"""
    KIS Data Loader

    @author: Younghyun Kim
    Created: 2022.10.03
"""
import pickle
import requests
import json
import datetime
import numpy as np
import pandas as pd

from pykrx import stock

from trading.account_info import KIS_ACCOUNT_INFO
from trading.constants import *
from trading.connection import get_access_token


class KISDataLoader:
    """
        KIS Data Loader
    """
    def __init__(self, access_token=None,
                app_key=None, app_secret=None):
        """
            Initialization
        """
        self.today = datetime.datetime.today().strftime("%Y%m%d")
        if app_key is None:
            self.app_key = KIS_ACCOUNT_INFO['app_key']

        if app_secret is None:
            self.app_secret = KIS_ACCOUNT_INFO['app_secret']

        if access_token is None:
            self.access_token = get_access_token()

        try:
            with open(TRADE_DATES_PATH, 'rb') as f:
                self.trade_dates = pickle.load(f)

            if self.today not in self.trade_dates:
                tdates_new = stock.get_previous_business_days(
                    fromdate=self.trade_dates[-1],
                    todate=self.today)

                if len(tdates_new[1:]) > 0:
                    for i, date in enumerate(tdates_new):
                        tdates_new[i] =\
                            date.to_pydatetime().strftime("%Y%m%d")

                        if i > 0:
                            self.trade_dates.append(tdates_new[i])

                    with open(TRADE_DATES_PATH, 'wb') as f:
                        pickle.dump(self.trade_dates, f)
        except Exception:
            self.trade_dates = None
            print("거래일 데이터가 존재하지 않음.")

    def get_timeseries(self, stock_code,
                start_date=None, end_date=None,
                period='D'):
        """
            Get Historical Stock Price
            Args:
                stock_code: 종목 코드
                start_date: start date
                end_date: end_date
                period
                    'D': Daily
                    'W': Weekly
                    'M': Monthly
                    'Y': Annually
            Return:
                timeseries_total: stock price series
                    * dtype: pd.DataFrame
                    * columns
                        open: 시가
                        high: 고가
                        low: 저가
                        close: 종가
                        volume: 거래량
                        turnover: 거래대금
        """
        if start_date is None:
            start_date = end_date

        # 거래일 조회
        if self.trade_dates is None:
            trade_dates = stock.get_previous_business_days(
                fromdate=start_date, todate=end_date)

            for i, date in enumerate(trade_dates):
                trade_dates[i] =\
                    date.to_pydatetime().strftime("%Y%m%d")
        else:
            start_idx = self.trade_dates.index(start_date)
            end_idx = self.trade_dates.index(end_date)
            trade_dates = self.trade_dates[start_idx:end_idx+1]

        req_num = len(trade_dates) // 100
        if len(trade_dates) % 100 != 0:
            req_num += 1

        for i in range(req_num):
            date_rng = trade_dates[i*100:(i+1)*100]
            start_date = date_rng[0]
            end_date = date_rng[-1]

            timeseries = self._get_timeseries(
                stock_code, start_date, end_date, period=period)

            if i == 0:
                timeseries_total = timeseries
            else:
                timeseries_total = timeseries_total.append(timeseries)

        timeseries_total = timeseries_total[
            ~timeseries_total.index.duplicated(keep='first')]

        return timeseries_total

    def _get_timeseries(self, stock_code,
                start_date=None, end_date=None,
                period='D'):
        """
            Get Historical Stock Price - Limit 100

            Args:
                stock_code: 종목 코드
                start_date: start date
                end_date: end_date
                period
                    'D': Daily
                    'W': Weekly
                    'M': Monthly
                    'Y': Annually
            Return:
                timeseries: stock price series
                    * dtype: pd.DataFrame
                    * columns
                        open: 시가
                        high: 고가
                        low: 저가
                        close: 종가
                        volume: 거래량
                        turnover: 거래대금
        """
        url = f"{URL_BASE}/{GET_PRICE_PATH}"

        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appKey": self.app_key,
            "appSecret": self.app_secret,
            "tr_id": "FHKST03010100"
        }

        params = {
            'fid_cond_mrkt_div_code': "J",
            'fid_input_iscd': stock_code,
            'fid_input_date_1': start_date,
            'fid_input_date_2': end_date,
            'fid_period_div_code':  period,
            'fid_org_adj_prc': "0"
        }

        res = requests.get(url, headers=headers, params=params)

        rst = res.json()

        timeseries = pd.DataFrame(
            rst['output2']).sort_values('stck_bsop_date')
        timeseries = timeseries.set_index('stck_bsop_date')

        timeseries = timeseries.rename(
            columns={
                'stck_clpr': 'close',
                'stck_oprc': 'open',
                'stck_hgpr': 'high',
                'stck_lwpr': 'low',
                'acml_vol': 'volume',
                'acml_tr_pbmn': 'turnover'
            }
        )

        timeseries = timeseries[['open', 'high', 'low', 'close',
                    'volume', 'turnover']].astype(float)

        return timeseries

    def inquire_daily_price(self, stock_code, period="D"):
        """
            Inquire Daily Stock Price
        """
        url = f"{URL_BASE}/{INQUIRE_DAILY_PRICE_PATH}"

        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appKey": self.app_key,
            "appSecret": self.app_secret,
            "tr_id": "FHKST01010100"
        }

        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_input_iscd": stock_code,
            "fid_org_adj_prc": "1",
            "fid_period_div_code": period
        }

        res = requests.get(url, headers=headers, params=params)

        return res.json()

    def inquire_minute_bar(self, stock_code, target_time):
        """
            Inquire Minute Bar
        """
        url = f"{URL_BASE}/{INQUIRE_TIME_ITEMCHARTPRICE}"

        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appKey": self.app_key,
            "appSecret": self.app_secret,
            "tr_id": "FHKST03010200"
        }

        params = {
            'fid_etc_cls_code': "",
            'fid_cond_mrkt_div_code': "J",
            'fid_input_hour_1': target_time,
            'fid_input_iscd': stock_code,
            'fid_pw_data_incu_yn': "Y",
        }

        res = requests.get(url, headers=headers, params=params)

        return res.json()