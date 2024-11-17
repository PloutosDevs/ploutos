import pandas as pd
from tqdm import tqdm
import numpy as np
import sys


from source.utils import normalize_prices
from source.feature_engeneering.super_trend import calculate_supertrend
from source.feature_engeneering.macd import calculate_macd
from source.feature_engeneering.rsi import calculate_rsi
from source.feature_engeneering.obv import calculate_obv_to_volume_ratio
from source.feature_engeneering.bollinger_bands import distance_between_bb_bands
from source.feature_engeneering.chaikin_money_flow import calculate_cmf
from source.feature_engeneering.rate_of_change import calculate_price_rate_of_change, calculate_volume_rate_of_change
from source.feature_engeneering.volume_ratio import calculate_volume_ratio
from source.feature_engeneering.stoch_rsi import calculate_stoch_rsi
from source.feature_engeneering.linear_regression import calculate_trailing_linear_reg_params
from source.feature_engeneering.btc_features import calculate_btc_features
from source.feature_engeneering.fear_and_greed import calculate_fear_and_greed_index
from source.feature_engeneering.dominance import calculate_dominance
from source.feature_engeneering.fibonacci_level import calculate_fibonacci_levels
from source.feature_engeneering.volatility_functions import calculate_traling_atr
from source.feature_engeneering.sup_and_res_levels import get_breakdown_and_bounce_levels
from source.feature_engeneering.candles_patterns import get_finish_abnormal_moves, get_candle_patterns
from source.feature_engeneering.time_series_features import get_time_series_features


def create_lag_features(data, column_names, lag=1):
    """
    Create lag features for a specific column in a DataFrame.

    Args:
    - data: Pandas DataFrame containing the dataset
    - column_name: Name of the column for which lag features will be created
    - lag: Number of time steps to shift the column (default is 1)

    Returns:
    - DataFrame with added lag features
    """

    # Create a copy of the original DataFrame
    df = data.copy()

    for column_name in column_names:
        # Create lag features by shifting the column values
        for i in range(1, lag + 1):
            lag_series = df[column_name].shift(i)
            lag_series.name = f'{column_name}_lag_{i}'
            df = pd.concat([df, lag_series], axis=1)

    return df


def add_features(data: pd.DataFrame, exp_config) -> pd.DataFrame:

    symbol_features_cols = sum(
        list(map(lambda x: x[0], exp_config['features'].get("symbol_features", {}).values())), []
    )
    dates_features_cols = sum(
        list(map(lambda x: x[0], exp_config['features'].get("dates_features", {}).values())), []
    )
    feature_cols = symbol_features_cols + dates_features_cols

    upd_data = []

    for symbol in tqdm(data["Symbol"].unique(), file=sys.stdout):
        
        # Get sample of prices by ticker
        sample = data[data["Symbol"] == symbol].copy()
        
        if sample.shape[0] < 10:
            continue

        # Calculate trailing cumulative yield over specified rolling window
        # Calculate yield at 1 day before cumulative yield. In other words yield_before_pump is not included in calculation of cum_prod, yield_before_pump == before pump date
        sample.loc[:, "cum_prod"] = (
            sample["Close"].pct_change()
                           .add(1)
                           .rolling(window=exp_config['strategy_params']["validation_window"])
                           .apply(lambda x: x.cumprod().max())  # np.prod
                           .subtract(1)
                           .multiply(100)
                           .fillna(0)
                           .shift(- exp_config['strategy_params']["validation_window"])  # Приводим доходность к дате начала пампа
                           .values
        )
        sample.loc[:, "yield_before_pump"] = (
            sample["Close"].pct_change().multiply(100)
        )

        # Calculate features relates to symbol
        for feature, values in exp_config['features'].get("symbol_features", {}).items():
            sample = eval(feature)(sample, **values[1])

        upd_data.append(sample.copy())

    upd_data = pd.concat(upd_data)

    # Calculate features relates to dates
    for feature, values in exp_config['features'].get("dates_features", {}).items():
        upd_data = eval(feature)(upd_data, **values[1])

    data_set = []

    # Calculate lags
    for symbol in tqdm(upd_data["Symbol"].unique()):

        sample = upd_data[upd_data["Symbol"] == symbol]

        # Important: ffill berfore bfill!!! Won't look at future
        feature_df = sample.ffill().bfill()
        feature_df = create_lag_features(
            feature_df, feature_cols, lag=exp_config['strategy_params']['last_features_window']
        )

        data_set.append(feature_df.copy())

    data_set = pd.concat(data_set)

    return data_set
