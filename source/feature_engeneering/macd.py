
def calculate_macd(prices_df, short_period=12, long_period=26, smoothing_period=9):
    """
    Receive DataFrame with prices and calculate MACD indicator. Add values in original DataFrame.

    params:
        prices_df - High, Low, Close, Open, Volume values
        short_period - smoothing period for short ema
        long_period - smoothing period for long ema
        smoothing_period - smoothing period for signal line
    return:
        Add in original DataFrame new cols "MACD", "Signal_Line", "Bar_Charts"
    """

    short_ema = prices_df['Close'].ewm(span=short_period, adjust=False).mean()
    long_ema = prices_df['Close'].ewm(span=long_period, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=smoothing_period, adjust=False).mean()
    bars = macd - signal_line

    prices_df['MACD'] = macd
    prices_df['MACD_Signal_Line'] = signal_line
    prices_df['MACD_Bar_Charts'] = bars

    return
