import re
from io import StringIO

import pandas as pd
import requests

from utility.folder_creator import folder_creator

TIMEOUT = 2
CRUMB_LINK = 'https://finance.yahoo.com/quote/{0}/history?p={0}'
CRUMBLE_REGEX = r'CrumbStore":{"crumb":"(.*?)"}'
QUOTE_LINK = 'https://query1.finance.yahoo.com/v7/finance/download/{quote}?period1={dfrom}&period2={dto}&interval=1d&events=history&crumb={crumb}'


def yahoo_finance_history(symbol, startdate, enddate):
    return get_quote(symbol, requests.session(), startdate, enddate)


def get_crumb(symbol):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}
    response = requests.get(CRUMB_LINK.format(symbol), timeout=TIMEOUT, headers=headers)
    response.raise_for_status()
    match = re.search(CRUMBLE_REGEX, response.text)
    if not match:
        raise ValueError('Could not get crumb from Yahoo Finance')
    else:
        crumb = match.group(1)
    return crumb


def get_quote(symbol, session, startdate, enddate):
    # if not hasattr(self, 'crumb') or len(session.cookies) == 0:
    crumb = get_crumb(symbol)
    datefrom = int(startdate.timestamp())
    dateto = int(enddate.timestamp())
    url = QUOTE_LINK.format(quote=symbol, dfrom=datefrom, dto=dateto, crumb=crumb)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text), parse_dates=['Date'])


def get_most_important_cryptos(cryptocurrencies, startdate, enddate):
    DATASET_NAME = "original"
    folder_creator("../acquisition/dataset", 0)
    DATASET_DIR = "../acquisition/dataset/" + DATASET_NAME
    folder_creator(DATASET_DIR, 1)
    currency = "-USD"
    # f = open("/crypto_symbols.txt", "r")
    # cryptos = f.readlines()
    for crypto in cryptocurrencies:
        # crypto = crypto.replace("\n", "")
        print("getting info about " + crypto)
        df = yahoo_finance_history(crypto + currency, startdate, enddate)
        df.to_csv(DATASET_DIR + "/" + crypto + ".csv", index=False)
