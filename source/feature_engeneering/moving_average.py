import pandas as pd

def calculate_ema(prices_df: pd.DataFrame, period: int, is_ratios: bool = False) -> pd.DataFrame:
    """
    Receive DataFrame with prices and calculate EMA indicator

    params:
        prices_df - High, Low, Close, Open, Volume values
        period - smoothing period for moving average
        is_ratios - flag for returning values as ratios to close price
    return:
        Return original DataFrame with new col "EMA"
    """

    prices_df['EMA'] = prices_df['Close'].ewm(span=period, adjust=False).mean()

    if is_ratios:
        prices_df['EMA'] = prices_df['EMA'].divide(prices_df['Close'])

    return prices_df


def calculate_sma(prices_df: pd.DataFrame, period: int, is_ratios: bool = False) -> pd.DataFrame:
    """
    Receive DataFrame with prices and calculate SMA indicator

    params:
        prices_df - High, Low, Close, Open, Volume values
        period - smoothing period for moving average
        is_ratios - flag for returning values as ratios to close price
    return:
        Return original DataFrame with new col "SMA"
    """

    prices_df['SMA'] = prices_df['Close'].rolling(window=period).mean()

    if is_ratios:
        prices_df['SMA'] = prices_df['SMA'].divide(prices_df['Close'])

    return prices_df
