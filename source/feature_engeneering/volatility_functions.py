import pandas as pd


def calculate_traling_std(prices_df, lookback):
    """
    Receive DataFrame with prices and calculate trailing STD. Add values in original DataFrame.

    params:
        prices_df - High, Low, Close, Open, Volume values
        lookback - period for calculating volatility
    return:
        Add in original DataFrame new col "Trailing_STD"
    """

    prices_df["Trailing_STD"] = prices_df["Close"].rolling(window=lookback).std().bfill()

    return


def calculate_traling_atr(prices_df, lookback):
    """
    Receive DataFrame with prices and calculate trailing ATR. Add values in original DataFrame.

    params:
        prices_df - High, Low, Close, Open, Volume values
        lookback - period for calculating volatility
    return:
        Add in original DataFrame new col "Trailing_ATR"
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

    return
