import pandas as pd
import requests
import numpy as np
from time import sleep
from tqdm import tqdm
import config

from source.utils import get_time_slide_window


def get_binance_symbols(only_usdt=True) -> np.array:
    """Get names of all binance symbols"""

    exchange_info = requests.get("https://api.binance.com/api/v3/exchangeInfo").json()
    
    all_symbols = pd.DataFrame(exchange_info['symbols'])
    
    if only_usdt:
        symbols_with_usdt = all_symbols[all_symbols['symbol'].str.contains("USDT")]['symbol'].unique()
        return symbols_with_usdt
    else:
        return all_symbols['symbol'].unique()


def get_candles_spot_binance(symbol: str, interval: str, start_time: str, end_time=None,
                             time_zone="Europe/Moscow", limit=500) -> pd.DataFrame:
    """
    Return candles of spot pairs from Binance exchange according to params.

    params:
        symbol: Symbol of currency pair, for example "BTCUSDT"
        interval: Interval of candles. All values can be seen in utils.get_time_slide_window
        start_time: Start datetime of candles in format "%Y-%d-%m %H:%M:%S.%f"
        end_time: End datetime of candles in format "%Y-%d-%m %H:%M:%S.%f". Default is now
        time_zone: Your timezone pytz.all_timezones
    return:
        Candles with OHLCV cols and pd.Timestamp index
    """

    # Binance get time in terms of ms and in UTC
    start_time = pd.Timestamp(start_time, tz=time_zone).tz_convert(tz="UTC").timestamp() * 1000

    # If end time is not limited, binance can return last prices infinitely
    if not end_time or pd.Timestamp(end_time, tz=time_zone) > pd.Timestamp.now(tz=time_zone):
        end_time = pd.Timestamp.now(tz=time_zone).tz_convert(tz="UTC").timestamp() * 1000
    else:
        end_time = pd.Timestamp(end_time, tz=time_zone).tz_convert(tz="UTC").timestamp() * 1000

    # Base DF
    base_candles = pd.DataFrame({}, columns=["Time", "Open", "High", "Low", "Close", "Volume"]).astype(float)

    # Define periods for requests to get effective number of candles according to interval and limits on return
    min_add_time = get_time_slide_window(interval, "binance", limit=limit)

    intervals = np.linspace(start_time, end_time, int((end_time - start_time) / min_add_time) + 1, endpoint=False,
                            retstep=True)

    if intervals[1] > min_add_time:
        raise ValueError("Incorrect interval")

    for start_time in intervals[0]:

        body = {"symbol": symbol, "interval": interval, "startTime": int(start_time), "limit": limit}

        # Try requests 3 times before status code == 200
        for _ in range(3):
            response = requests.get("https://api.binance.com/api/v3/klines", body)

            if response.status_code == 200:
                response = response.json()
                break

        if isinstance(response, requests.Response):
            raise requests.exceptions.ConnectionError(response.text)

        if response:
            candles = pd.DataFrame(response)
            candles = candles[candles.columns[:6]]
            candles.columns = ["Time", "Open", "High", "Low", "Close", "Volume"]

            candles['Time'] = candles["Time"].apply(
                lambda x: pd.Timestamp.fromtimestamp(x / 1000, tz="UTC").tz_convert(tz=time_zone)
            )

            base_candles = pd.concat([base_candles if not base_candles.empty else None, candles])

        if start_time >= end_time:
            break

    base_candles = base_candles.drop_duplicates("Time").set_index("Time").astype(float)

    base_candles = base_candles.loc[:pd.Timestamp.fromtimestamp(end_time / 1000, tz="UTC").tz_convert(tz=time_zone)]

    return base_candles


def compose_binance_candles_df(symbols: list, start_time: str, end_time: str = None):

    results_df = pd.DataFrame(columns=["Time", "Open", "High", "Low", "Close", "Volume", "Symbol"]).set_index("Time")
    
    for symbol in tqdm(symbols):

        try:
            df = get_candles_spot_binance(symbol, "1d", start_time=start_time, end_time=end_time,
                                          time_zone=config.DEFAULT_TZ)
            if not df.empty:
                df.loc[:, "Symbol"] = symbol
                results_df = pd.concat([results_df if not results_df.empty else None, df])
        except ConnectionError:
            sleep(10)
            df = get_candles_spot_binance(symbol, "1d", start_time=start_time, end_time=end_time,
                                          time_zone=config.DEFAULT_TZ)
            if not df.empty:
                df.loc[:, "Symbol"] = symbol
                results_df = pd.concat([results_df if not results_df.empty else None, df])

    return results_df
