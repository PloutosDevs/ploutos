import requests
import pandas as pd

import config

def calculate_fear_and_greed_index(candels_df):
    """
    Receives DataFrame with prices and calculates Fear and Greed Index

    params:
        candles_df - High, Low, Close, Open, Volume values
    return:
        Original DataFrame new col fear_and_greed
    """
    response = requests.get("https://api.alternative.me/fng/?limit=0&format=json&date_format=world").json()

    fng = pd.DataFrame(response["data"]).set_index("timestamp")["value"].to_frame("fear_and_greed").astype(float)
    fng.index = pd.DatetimeIndex(fng.index, tz="UTC").tz_convert(config.DEFAULT_TZ)

    candels_df = candels_df.merge(fng, right_index=True, left_index=True, how="left")

    return candels_df
