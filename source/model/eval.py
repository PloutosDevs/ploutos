from source.data.get.binance_prices import get_binance_symbols, compose_binance_candles_df
from source.data.process.compose_features import add_features
from joblib import load
import os
import config
from source import utils
import pandas as pd


def eval_model(valuation_date: pd.Timestamp = pd.Timestamp.today(tz=config.DEFAULT_TZ).normalize(),
               candles_period: int = 60, best_symbols_offset: int = 50, yield_before_pump: float = 0.03):

    # Get data
    print('Start to get candels')
    BINANCE_SYMBOLS = get_binance_symbols(only_usdt=True)
    # BINANCE_SYMBOLS = ['LOOMUSDT', 'HIFIUSDT', 'VICUSDT', 'RPLUSDT', 'ATAUSDT']

    candles_start_date = (valuation_date - pd.Timedelta(days=candles_period)).strftime("%Y-%m-%d")
    candles_end_date = valuation_date.strftime("%Y-%m-%d") + " 23:59:59.999999"

    eval_candles_df = compose_binance_candles_df(BINANCE_SYMBOLS, start_time=candles_start_date,
                                                 end_time=candles_end_date)
    
    EXPERIMENT_CONFIG = {
        'features': {
            "calculate_supertrend": [["SuperTrend"], {"vol_func": "atr", "period": 20, "multiplier": 2.5}],
            "calculate_macd": [
                ["MACD_Signal_Line", "MACD", "MACD_Bar_Charts"],
                {"short_period": 12, "long_period": 26, "smoothing_period": 9}
            ],
            "calculate_rsi": [["RSI"], {"period": 20, "ema": True}],
            "calculate_obv_to_volume_ratio": [["OBV_Volume_Ratio"], {}],
            "distance_between_bb_bands": [
                ["Upper_distance", "Lower_distance"],
                {"period": 20, "multiplier": 2.5, "ema": 2.5, "normalize": True}
            ],
            "calculate_cmf": [["CMF"], {"period": 20}],
            "calculate_price_rate_of_change": [["Price_ROC"], {}],
            "calculate_volume_rate_of_change": [["Volume_ROC"], {}],
            "calculate_volume_ratio": [["Volume_Ratio"], {"period": 20, "ema": True}],
            "calculate_stoch_rsi": [
                ["Stoch_RSI_K", "Stoch_RSI_D"],
                {"rsi_period": 20, "k_period": 20, "smooth_k": 5, "smooth_k": 5, "ema": True}
            ],
            "calculate_trailing_linear_reg_params": [["Reg_Coef", "RMSE"], {"period": 25, "col_name": "cum_prod"}]
        },
        'strategy_params': {
            "last_features_window": 7,
            "candles_between_pump": 30,
            "validation_window": 20,  # candles
            "min_yield": -20,  # %
            "max_yield": 20,
            "first_yield": 3
        }
    }
    
    
    # Generate features
    print('Start to generate features')
    eval_features_df = add_features(eval_candles_df, exp_config=EXPERIMENT_CONFIG)

    # Load model
    model = load(os.path.join(config.MODELS_PATH, "xgb_model_new.joblib"))

    # Predict for today
    print('Start to predict')
    eval_features_df_today = eval_features_df.loc[eval_features_df.index.max()].copy()

    eval_features_df_today.loc[:, 'proba'] = model.best_estimator_.predict_proba(
        eval_features_df_today.loc[:, model.feature_names_in_]
    )[:, 1]

    eval_features_df_today.loc[:, 'predict'] = model.best_estimator_.predict(
        eval_features_df_today.loc[:, model.feature_names_in_]
    )
    
    # Find best symbols
    best_symbols = (
        eval_features_df_today.loc[eval_features_df_today.predict == 1, ['Symbol', 'predict', 'proba']]
                              .sort_values('proba', ascending=False)
                              .head(best_symbols_offset)
    )
    
    # Make plots
    plots_buffer = utils.create_plot_best_symbols(eval_candles_df, best_symbols, show=False)
    
    return plots_buffer, best_symbols.Symbol.values.tolist()
