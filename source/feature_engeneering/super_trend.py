import pandas as pd
import numpy as np

from source.feature_engeneering.volatility_functions import calculate_traling_atr, calculate_traling_std


def calculate_supertrend(
        prices_df: pd.DataFrame, vol_func: str, period:  int, multiplier: float, is_ratios: bool = False
) -> pd.DataFrame:
    """
    Receive DataFrame with prices and calculate super trend indicator

    params:
        prices_df - High, Low, Close, Open, Volume values
        vol_func - volatility function. Can be: std, atr
        period - period for calculating volatility
        multiplier - coefficient for defining distance between price and bands
        is_ratios - flag for returning values as ratios to close price
    return:
        Return original DataFrame with new col "SuperTrend"
    """

    high = prices_df['High']
    low = prices_df['Low']
    close = prices_df['Close']

    if vol_func == "std":
        calculate_traling_std(prices_df, period)
        vol = prices_df['Trailing_STD']
    elif vol_func == "atr":
        calculate_traling_atr(prices_df, period)
        vol = prices_df['Trailing_ATR']

    hl_avg = (high + low) / 2
    upper_band = (hl_avg + multiplier * vol).dropna()
    lower_band = (hl_avg - multiplier * vol).dropna()

    # FINAL UPPER BAND
    final_bands = pd.DataFrame(columns=['upper', 'lower'])
    final_bands.iloc[:, 0] = [x for x in upper_band - upper_band]
    final_bands.iloc[:, 1] = final_bands.iloc[:, 0]
    for i in range(len(final_bands)):
        if i == 0:
            final_bands.iloc[i, 0] = 0
        else:
            if (upper_band.iloc[i] < final_bands.iloc[i - 1, 0]) | (close.iloc[i - 1] > final_bands.iloc[i - 1, 0]):
                final_bands.iloc[i, 0] = upper_band.iloc[i]
            else:
                final_bands.iloc[i, 0] = final_bands.iloc[i - 1, 0]

    # FINAL LOWER BAND
    for i in range(len(final_bands)):
        if i == 0:
            final_bands.iloc[i, 1] = 0
        else:
            if (lower_band.iloc[i] > final_bands.iloc[i - 1, 1]) | (close.iloc[i - 1] < final_bands.iloc[i - 1, 1]):
                final_bands.iloc[i, 1] = lower_band.iloc[i]
            else:
                final_bands.iloc[i, 1] = final_bands.iloc[i - 1, 1]

    # SUPERTREND
    supertrend = pd.DataFrame(columns=['supertrend'])
    supertrend.iloc[:, 0] = [x for x in final_bands['upper'] - final_bands['upper']]

    for i in range(len(supertrend)):
        if i == 0:
            supertrend.iloc[i, 0] = 0
        elif supertrend.iloc[i - 1, 0] == final_bands.iloc[i - 1, 0] and close.iloc[i] < final_bands.iloc[i, 0]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 0]
        elif supertrend.iloc[i - 1, 0] == final_bands.iloc[i - 1, 0] and close.iloc[i] > final_bands.iloc[i, 0]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
        elif supertrend.iloc[i - 1, 0] == final_bands.iloc[i - 1, 1] and close.iloc[i] > final_bands.iloc[i, 1]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
        elif supertrend.iloc[i - 1, 0] == final_bands.iloc[i - 1, 1] and close.iloc[i] < final_bands.iloc[i, 1]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 0]

    supertrend = supertrend.set_index(upper_band.index)
    supertrend = supertrend.dropna()[1:]

    # ST UPTREND/DOWNTREND
    upt = []
    dt = []
    close = close.iloc[len(close) - len(supertrend):]

    for i in range(len(supertrend)):
        if close.iloc[i] > supertrend.iloc[i, 0]:
            upt.append(supertrend.iloc[i, 0])
            dt.append(np.nan)
        elif close.iloc[i] < supertrend.iloc[i, 0]:
            upt.append(np.nan)
            dt.append(supertrend.iloc[i, 0])
        else:
            upt.append(np.nan)
            dt.append(np.nan)

    st, upt, dt = pd.Series(supertrend.iloc[:, 0]), pd.Series(upt), pd.Series(dt)
    upt.index, dt.index = supertrend.index, supertrend.index
    prices_df['SuperTrend'] = st.bfill().ffill()

    if is_ratios:
        prices_df['SuperTrend'] = prices_df['SuperTrend'].divide(prices_df['Close'])

    return prices_df
