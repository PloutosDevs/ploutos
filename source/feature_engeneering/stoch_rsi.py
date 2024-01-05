from source.feature_engeneering.rsi import calculate_rsi


def calculate_stoch_rsi(prices_df, rsi_period=14, k_period=14, smooth_k=3, smooth_d=3, ema=False):
    """
    Receive DataFrame with prices and calculate Stoch RSI indicator. Add values in original DataFrame.

    params:
        prices_df - High, Low, Close, Open, Volume values
        vol_func - volatility function. Can be: std, atr
        lookback - period for calculating volatility
        multiplier - coefficient for defining distance between price and bands
    return:
        Add in original DataFrame new cols "Stoch_RSI_K", "Stoch_RSI_D"
    """

    new_prices_df = prices_df.copy()
    calculate_rsi(new_prices_df, rsi_period, ema)

    new_prices_df['n_high'] = new_prices_df['RSI'].rolling(k_period).max()
    new_prices_df['n_low'] = new_prices_df['RSI'].rolling(k_period).min()
    new_prices_df['%K'] = (
            (new_prices_df['RSI'] - new_prices_df['n_low']) * 100 /
            (new_prices_df['n_high'] - new_prices_df['n_low'])
    )

    if ema:
        k = new_prices_df['%K'].ewm(span=smooth_k, adjust=False).mean()
        prices_df[f"Stoch_RSI_K"] = k
        prices_df[f"Stoch_RSI_D"] = k.ewm(span=smooth_d, adjust=False).mean()
    else:
        k = new_prices_df['%K'].rolling(smooth_k).mean()
        prices_df['Stoch_RSI_K'] = k
        prices_df['Stoch_RSI_D'] = k.rolling(smooth_d).mean()

    return
