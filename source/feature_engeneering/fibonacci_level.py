import pandas as pd


def calculate_fibonacci_levels(candles_df: pd.DataFrame, period: int, type_deal: str) -> pd.DataFrame:
    """
    Receives DataFrame with prices and calculates Close prices to Fibonacci levels ratio for period.

    params:
        candles_df - High, Low, Close, Open, Volume values
        period - period for price range
        type_deal - "long" or "short"
    return:
        Original DataFrame new cols fibo_23.6, fibo_38.2, fibo_50, fibo_61.8, fibo_100
    """

    fibonacci_levels = {
        '23.6': 0.236,
        '38.2': 0.382,
        '50': 0.5,
        '61.8': 0.618,
        '100': 1
    }

    for sample in candles_df.rolling(window=period):

        if len(sample) < period:
            continue

        last_close = sample.loc[sample.index[-1], "Close"]
        max_price = sample["High"].max()
        min_price = sample["Low"].min()
        price_range = max_price - min_price

        for level, value in fibonacci_levels.items():

            if type_deal == "long":
                candles_df.loc[sample.index[-1], "fibo_" + level] = last_close / (price_range * value + min_price)
            elif type_deal == "short":
                candles_df.loc[sample.index[-1], "fibo_" + level] = (max_price - price_range * value) / last_close
            else:
                raise ValueError(f"There isn't {type_deal} deal type")

    return candles_df
