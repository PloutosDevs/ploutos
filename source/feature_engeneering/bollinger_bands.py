from source.utils import min_max_normalization


def calculate_bb_bands(prices_df, period, multiplier, ema=True):
    """
    Receive DataFrame with prices and calculate Bollinger bands indicator

    params:
        prices_df - High, Low, Close, Open, Volume values
        period - smoothing period for moving average
        multiplier - coefficient for defining distance between price and bands
        ema - use ema or sma
    return:
        Return original DataFrame with new cols "Upper_Band", "Lower_Band"
    """

    if ema:
        prices_df['BB_MA'] = prices_df['Close'].ewm(span=period, adjust=False).mean()
    else:
        prices_df['BB_MA'] = prices_df['Close'].rolling(window=period).mean()

    prices_df['Upper_Band'] = prices_df['BB_MA'] + multiplier * prices_df['Close'].rolling(window=period).std()
    prices_df['Lower_Band'] = prices_df['BB_MA'] - multiplier * prices_df['Close'].rolling(window=period).std()

    prices_df.drop("BB_MA", axis=1, inplace=True)

    return prices_df


def distance_between_bb_bands(prices_df, period, multiplier, ema=True, normalize=False):
    """
    Receive DataFrame with prices, calculate distance between Bollinger bands and Close.

    params:
        prices_df - High, Low, Close, Open, Volume values
        period - smoothing period for moving average
        multiplier - coefficient for defining distance between price and bands
        ema - use ema or sma
        normalize - apply min max normalization
    return:
        Return original DataFrame with new cols "Upper_distance", "Lower_distance"
    """
    new_prices_df = prices_df.copy()

    calculate_bb_bands(new_prices_df, period, multiplier, ema=ema)

    new_prices_df["Upper_distance"] = new_prices_df["Upper_Band"] - new_prices_df["Close"]
    new_prices_df["Lower_distance"] = new_prices_df["Close"] - new_prices_df["Lower_Band"]

    prices_df["Upper_distance"] = new_prices_df["Upper_distance"]
    prices_df["Lower_distance"] = new_prices_df["Lower_distance"]

    if normalize:
        prices_df["Upper_distance"] = min_max_normalization(prices_df["Upper_distance"])
        prices_df["Lower_distance"] = min_max_normalization(prices_df["Lower_distance"])

    return prices_df
