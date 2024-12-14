import pandas as pd
from tvDatafeed import TvDatafeed, Interval

pd.set_option('display.width', None)
import time

username = 'YourTradingViewUsername'
password = 'YourTradingViewPassword'

tv = TvDatafeed(username, password)
symbol = "KCB"

def get_data(symbol):
    data = tv.get_hist(symbol=symbol, exchange='NSEKE', interval=Interval.in_daily, n_bars=8000)
    csv_data = data.to_csv(f"{symbol}.csv")
    return csv_data

get_data(symbol)


