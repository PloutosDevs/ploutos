
def calculate_ema(prices_df, period):
    """
    Receive DataFrame with prices and calculate EMA indicator

    params:
        prices_df - High, Low, Close, Open, Volume values
        period - smoothing period for moving average
    return:
        Return original DataFrame with new col "EMA"
    """

    prices_df['EMA'] = prices_df['Close'].ewm(span=period, adjust=False).mean()

    return prices_df


def calculate_sma(prices_df, period):
    """
    Receive DataFrame with prices and calculate SMA indicator

    params:
        prices_df - High, Low, Close, Open, Volume values
        period - smoothing period for moving average
    return:
        Return original DataFrame with new col "SMA"
    """

    prices_df['SMA'] = prices_df['Close'].rolling(window=period).mean()

    return prices_df
