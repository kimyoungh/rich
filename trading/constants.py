"""
    Constants for KIS

    @author: Younghyun Kim
    Created: 2022.10.04
"""
URL_BASE = "https://openapi.koreainvestment.com:9443"
OAUTH2_PATH = "oauth2/tokenP"
INQUIRE_DAILY_PRICE_PATH =\
    "uapi/domestic-stock/v1/quotations/inquire-daily-price"
GET_PRICE_PATH=\
    "uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
INQUIRE_TIME_ITEMCHARTPRICE =\
    "uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"

# Trade Dates Path
TRADE_DATES_PATH = "./trading_data/trade_dates.pkl"