import os
import pandas as pd

from source.data.get.coin_market_cap import CoinMarketCap

import config

# Из файла
# def calculate_dominance(candles_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Receives DataFrame with prices and calculates BTC and ETH dominance
#
#     params:
#         candles_df - High, Low, Close, Open, Volume values
#     return:
#         Original DataFrame new cols btc_dominance and eth_dominance
#     """
#
#     btc_dominance = pd.read_csv(os.path.join(config.DATA_PATH, "btc_dominance.csv"), sep=";").set_index("DateTime")
#
#     btc_dominance.index = pd.DatetimeIndex(btc_dominance.index, tz="utc").normalize()
#     btc_dominance = btc_dominance.apply(lambda x: x.str.replace(",", ".")).astype(float)
#
#     btc_dominance = btc_dominance.apply(lambda col: col.divide(btc_dominance.sum(axis=1)))[["BTC", "ETH"]].rename(
#         columns={"BTC": "btc_dominance", "ETH": "eth_dominance"}
#     )
#
#     candles_df = candles_df.merge(btc_dominance, right_index=True, left_index=True, how="left")
#
#     return candles_df

def calculate_dominance(candles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Receives DataFrame with prices and calculates BTC and ETH dominance

    params:
        candles_df - High, Low, Close, Open, Volume values
        coin - Class instance for working with CoinMarketCap API
    return:
        Original DataFrame new cols btc_dominance and eth_dominance
    """

    coin = CoinMarketCap()

    time_start = candles_df.index[0].strftime("%Y-%m-%d")
    time_end = candles_df.index[-1].strftime("%Y-%m-%d")

    params = {
        "time_start": time_start,
        "time_end": time_end,
        "interval": "7d",
        "aux": "btc_dominance,eth_dominance"
    }
    res = coin.send_request("GET", "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/historical",
                            params=params)

    btc_dominance = pd.json_normalize(res["data"]["quotes"]).set_index("timestamp")[["btc_dominance", "eth_dominance"]]
    btc_dominance.index = pd.DatetimeIndex(btc_dominance.index).normalize()

    candles_df = candles_df.merge(btc_dominance, right_index=True, left_index=True, how="left")

    return candles_df
