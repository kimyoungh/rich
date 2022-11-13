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

def get_hashkey(asset='stock'):
    """
        get hashkey

        Args:
            asset
                * stock
                * deriv
                * default: stock
    """
    if asset == 'stock':
        cano = KIS_ACCOUNT_INFO['dom_stk_acnt']
        acnt_prdt_cd = KIS_ACCOUNT_INFO['dom_stk_acnt_prdt_cd']
    else:
        cano = KIS_ACCOUNT_INFO['dom_deri_acnt']
        acnt_prdt_cd = KIS_ACCOUNT_INFO['dom_deri_acnt_prdt_cd']

    datas = {
        'CANO': cano,
        'ACNT_PRDT_CD': acnt_prdt_cd,
        'OVRS_EXCG_CD': 'SHAA',
        'PDNO': '00001',
        'ORD_QTY': '500',
        'OVRS_ORD_UNPR': '52.65',
        'ORD_SVR_DVSN_CD': '0',
    }

    headers = {
        'content-Type': 'application/json',
        'appkey': KIS_ACCOUNT_INFO['app_key'],
        'appsecret': KIS_ACCOUNT_INFO['app_secret']}

    path = 'uapi/hashkey'
    url = f"{URL_BASE}/{path}"

    res = requests.post(url, headers=headers, data=json.dumps(datas))
    hashkey = res.json()['HASH']

    return hashkey