
def calculate_volume_ratio(prices_df, period, ema=True):
    """
    Receive DataFrame with prices and calculate volume ratio indicator. Add values in original DataFrame.

    params:
        prices_df - High, Low, Close, Open, Volume values
        period - smoothing period for moving average
        ema - use ema or sma
    return:
        Add in original DataFrame new col "Volume_Ratio"
    """
    if ema:
        prices_df['Volume_Ratio'] = prices_df['Volume'] / prices_df['Volume'].ewm(span=period, adjust=False).mean()
    else:
        prices_df['Volume_Ratio'] = prices_df['Volume'] / prices_df['Volume'].rolling(window=period).mean()

    return
