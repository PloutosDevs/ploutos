import pandas as pd


def calculate_traling_std(prices_df: pd.DataFrame, lookback: int, is_ratios: bool = False) -> pd.DataFrame:
    """
    Receive DataFrame with prices and calculate trailing STD

    params:
        prices_df - High, Low, Close, Open, Volume values
        lookback - period for calculating volatility
        is_ratios - flag for returning values as ratios to close price
    return:
        Return original DataFrame with new col "Trailing_STD"
    """

    prices_df["Trailing_STD"] = prices_df["Close"].rolling(window=lookback).std().bfill()

    if is_ratios:
        prices_df["Trailing_STD"] = prices_df["Trailing_STD"].divide(prices_df["Close"])

    return prices_df


def calculate_traling_atr(prices_df: pd.DataFrame, lookback: int, is_ratios: bool = False) -> pd.DataFrame:
    """
    Receive DataFrame with prices and calculate trailing ATR

    params:
        prices_df - High, Low, Close, Open, Volume values
        lookback - period for calculating volatility
        is_ratios - flag for returning values as ratios to close price
    return:
        Return original DataFrame with new col "Trailing_ATR"
    """

    high = prices_df['High']
    low = prices_df['Low']
    close = prices_df['Close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    frameworks = [tr1, tr2, tr3]
    tr = pd.concat(frameworks, axis=1, join='inner').max(axis=1)
    atr = tr.ewm(span=lookback, adjust=False).mean()

    prices_df['Trailing_ATR'] = atr

    if is_ratios:
        prices_df["Trailing_ATR"] = prices_df["Trailing_ATR"].divide(prices_df["Close"])

    return prices_df
