import numpy as np
import pandas as pd


def hurst_exponent(time_series):
    num_lags = min(20, len(time_series) // 2)
    lags = range(2, num_lags)
    tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0


def fractal_dimension(time_series):
    n = len(time_series)
    length = []
    for k in range(2, n):
        scale = n // k
        series_scaled = [np.mean(time_series[i*scale:(i+1)*scale]) for i in range(k)]
        length.append(np.std(series_scaled) / (scale ** 0.5))
    poly = np.polyfit(np.log(range(2, n)), np.log(length), 1)
    return -poly[0]


def get_time_series_features(candles_df, window=30):

    candles_df = candles_df.copy()

    for roll in candles_df['Close'].rolling(window):

        if len(roll) != window:
            continue

        candles_df.loc[roll.index[-1], 'hurst_exponent'] = hurst_exponent(roll.values)
        candles_df.loc[roll.index[-1], 'fractal_dimension'] = fractal_dimension(roll.values)

    return candles_df
