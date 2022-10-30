"""
    KIS Get module for access token

    @author: Younghyun Kim
    Created: 2022.10.03
"""
import requests
import json
from trading.constants import *
from trading.account_info import KIS_ACCOUNT_INFO

def get_access_token():
    """
        get access token
    """
    headers = {"content-type": "application/json"}
    body = {
        "grant_type": "client_credentials",
        "appkey": KIS_ACCOUNT_INFO['app_key'],
        "appsecret": KIS_ACCOUNT_INFO['app_secret']
        }

    url = f"{URL_BASE}/{OAUTH2_PATH}"

    res = requests.post(url, headers=headers,
                    data=json.dumps(body))

    access_token = res.json()['access_token']

    return access_token