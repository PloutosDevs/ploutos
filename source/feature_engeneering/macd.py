import pandas as pd

def calculate_macd(
        prices_df: pd.DataFrame, short_period: int = 12, long_period: int = 26, smoothing_period: int = 9,
        is_ratios: bool = False
) -> pd.DataFrame:
    """
    Receive DataFrame with prices and calculate MACD indicator

    params:
        prices_df - High, Low, Close, Open, Volume values
        short_period - smoothing period for short ema
        long_period - smoothing period for long ema
        smoothing_period - smoothing period for signal line
        is_ratios - flag for returning values as ratios to close price
    return:
        Return original DataFrame with new cols "MACD", "Signal_Line", "Bar_Charts"
    """

    short_ema = prices_df['Close'].ewm(span=short_period, adjust=False).mean()
    long_ema = prices_df['Close'].ewm(span=long_period, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=smoothing_period, adjust=False).mean()
    bars = macd - signal_line

    prices_df['MACD'] = macd
    prices_df['MACD_Signal_Line'] = signal_line
    prices_df['MACD_Bar_Charts'] = bars

    if is_ratios:
        prices_df['MACD'] = prices_df['MACD'].divide(prices_df['Close'])
        prices_df['MACD_Signal_Line'] = prices_df['MACD_Signal_Line'].divide(prices_df['Close'])
        prices_df['MACD_Bar_Charts'] = prices_df['MACD_Bar_Charts'].divide(prices_df['Close'])

    return prices_df
