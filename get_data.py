"""
this script can be use to download all data for SNP 500
"""

import datetime
import logging
from pathlib import Path

from yahoo_fin import stock_info as si

log = logging.getLogger()
data_dir = Path('data')

str_now = datetime.datetime.now().strftime('%Y_%m_%d')
list_ticker_snp_500 = si.tickers_sp500()

for ticker in list_ticker_snp_500:
    log.warning(f'downloading {ticker}')
    try:
        df = si.get_data(ticker)
        df.to_csv(data_dir / f'{ticker}_{str_now}.csv')

    except Exception as e:
        print("Oops!", e.__class__, "occurred.")

        print("Next ticker.")
