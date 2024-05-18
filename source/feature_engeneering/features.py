from source.features.super_trend import calculate_supertrend
from source.features.macd import calculate_macd
from source.features.rsi import calculate_rsi
from source.features.obv import calculate_obv_to_volume_ratio
from source.features.bollinger_bands import distance_between_bb_bands
from source.features.chaikin_money_flow import calculate_cmf
from source.features.rate_of_change import calculate_price_rate_of_change, calculate_volume_rate_of_change
from source.features.volume_ratio import calculate_volume_ratio
from source.features.stoch_rsi import calculate_stoch_rsi
from source.features.linear_regression import calculate_trailing_linear_reg_params

calculate_supertrend(sample, **{"vol_func": "atr", "lookback": 20, "multiplier": 2.5})
calculate_macd(sample, **{"shor_period": 12, "long_period": 26, "smoothing_period": 9})
calculate_rsi(sample, **{"period": 20, "ema": True})
calculate_obv_to_volume_ratio(sample, **{})
distance_between_bb_bands(sample, **{"period": 20, "multiplier": 2.5, "ema": 2.5, "normalize": True})
calculate_cmf(sample, **{"period": 20})
calculate_price_rate_of_change(sample, **{})
calculate_volume_rate_of_change(sample, **{})
calculate_volume_ratio(sample, **{"period": "ema", "ema": True})
calculate_stoch_rsi(sample, **{"rsi_period": 20, "k_period": 20, "smooth_k": 5, "smooth_k": 5, "ema": True})
calculate_trailing_linear_reg_params(sample, **{"period": 25, "col_name": "cum_prod"})



class Features:

    def __init__(self):
        self.features = {}

    def add_supertrend(self, prices_df, **kwargs):
        calculate_supertrend(prices_df, **kwargs)
        self.features["SuperTrend"] = kwargs

    def add_macd(self, prices_df, **kwargs):
        calculate_macd(prices_df, **kwargs)
        self.features["MACD"] = kwargs
        self.features["MACD_Signal_Line"] = kwargs
        self.features["MACD_Bar_Charts"] = kwargs

    def add_rsi(self, prices_df, **kwargs):
        calculate_rsi(prices_df, **kwargs)
        self.features["RSI"] = kwargs

    def



    add_supertrend.__doc__ = calculate_supertrend.__doc__
    add_macd.__doc__ = calculate_macd.__doc__
    add_rsi.__doc__ = calculate_rsi.__doc__



