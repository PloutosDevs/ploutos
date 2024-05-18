
def calculate_rsi(prices_df, period=14, ema=True):
    """
    Receive DataFrame with prices and calculate RSI indicator.

    params:
        prices_df - High, Low, Close, Open, Volume values
        period - smoothing period for moving average
        ema - use ema or sma
    return:
        Return original DataFrame with new col "RSI"
    """

    close_delta = prices_df['Close'].diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)

    if ema:
        # Use exponential moving average
        ma_up = up.ewm(com=period - 1, adjust=False, min_periods=period).mean()
        ma_down = down.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window=period).mean()
        ma_down = down.rolling(window=period).mean()

    rsi = ma_up / ma_down
    rsi = 100 - (100 / (1 + rsi))

    prices_df['RSI'] = rsi

    return prices_df
