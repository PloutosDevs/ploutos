import numpy as np
import pandas as pd


def calculate_obv(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Receive DataFrame with prices and calculate obv indicator

    params:
        prices_df - High, Low, Close, Open, Volume values
    return:
        Return original DataFrame with new col "OBV"
    """

    prices_df['OBV'] = np.where(
        prices_df['Close'] > prices_df['Close'].shift(1),
        prices_df['Volume'],
        np.where(
            prices_df['Close'] < prices_df['Close'].shift(1),
            -prices_df['Volume'], 0
        )
    )

    prices_df['OBV'] = prices_df['OBV'].cumsum()

    return prices_df


def calculate_obv_to_volume_ratio(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Receive DataFrame with prices and calculate OBV to Volume Ratio indicator

    params:
        prices_df - High, Low, Close, Open, Volume values
    return:
        Return original DataFrame with new col "OBV_Volume_Ratio"
    """

    new_prices_df = prices_df.copy()

    new_prices_df = calculate_obv(new_prices_df)

    new_prices_df["OBV_Volume_Ratio"] = new_prices_df["OBV"] / new_prices_df["Volume"]

    prices_df["OBV_Volume_Ratio"] = new_prices_df["OBV_Volume_Ratio"]

    return prices_df
