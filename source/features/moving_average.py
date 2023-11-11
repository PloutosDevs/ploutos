
def calculate_ema(prices_df, period):
    """
    Receive DataFrame with prices and calculate EMA indicator. Add values in original DataFrame.

    params:
        prices_df - High, Low, Close, Open, Volume values
        period - smoothing period for moving average
    return:
        Add in original DataFrame new col "EMA"
    """

    prices_df['EMA'] = prices_df['Close'].ewm(span=period, adjust=False).mean()

    return


def calculate_sma(prices_df, period):
    """
    Receive DataFrame with prices and calculate SMA indicator. Add values in original DataFrame.

    params:
        prices_df - High, Low, Close, Open, Volume values
        period - smoothing period for moving average
    return:
        Add in original DataFrame new col "SMA"
    """

    prices_df['SMA'] = prices_df['Close'].rolling(window=period).mean()

    return
