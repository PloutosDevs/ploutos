import pandas as pd


def calculate_cmf(prices_df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Receive DataFrame with prices and calculate Chaikin Money Flow (CMF) indicator

    params:
        prices_df - High, Low, Close, Open, Volume values
        period - smoothing period for moving sum
    return:
        Return original DataFrame with new col "CMF"
    """

    mf_multiplier = (
            ((prices_df['Close'] - prices_df['Low']) - (prices_df['High'] - prices_df['Close'])) /
            (prices_df['High'] - prices_df['Low'])
    )

    mf_volume = mf_multiplier * prices_df['Volume']

    prices_df['MF_Multiplier'] = mf_multiplier
    prices_df['MF_Volume'] = mf_volume

    prices_df['CMF'] = (
            prices_df['MF_Volume'].rolling(window=period).sum() / prices_df['Volume'].rolling(window=period).sum()
    )

    prices_df.drop(["MF_Multiplier", "MF_Volume"], axis=1, inplace=True)

    return prices_df
