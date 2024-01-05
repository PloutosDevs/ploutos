from source.data.get.binance_prices import get_binance_symbols, compose_binance_candles_df
from datetime import date, timedelta
from source.data.process.compose_features import add_features
from joblib import load
import os
import config
from source import utils



def eval_model():

    # Get data
    print('Start to get candels')
    BINANCE_SYMBOLS = get_binance_symbols(only_usdt=True)
    # BINANCE_SYMBOLS = ['LOOMUSDT', 'HIFIUSDT', 'VICUSDT', 'RPLUSDT', 'ATAUSDT']
    DAYS_BEFORE = 30
    DATE_BEFORE_TODAY = (date.today() - timedelta(days=DAYS_BEFORE)).strftime("%Y-%m-%d")
    eval_candels_df = compose_binance_candles_df(BINANCE_SYMBOLS, start_time=DATE_BEFORE_TODAY)
    
    EXPERIMENT_CONFIG = {
    'features': {
        "calculate_supertrend": [["SuperTrend"], {"vol_func": "atr", "period": 20, "multiplier": 2.5}],
        "calculate_macd": [["MACD_Signal_Line", "MACD", "MACD_Bar_Charts"], {"short_period": 12, "long_period": 26, "smoothing_period": 9}],
        "calculate_rsi": [["RSI"], {"period": 20, "ema": True}],
        "calculate_obv_to_volume_ratio": [["OBV_Volume_Ratio"], {}],
        "distance_between_bb_bands": [["Upper_distance", "Lower_distance"], {"period": 20, "multiplier": 2.5, "ema": 2.5, "normalize": True}],
        "calculate_cmf": [["CMF"], {"period": 20}],
        "calculate_price_rate_of_change": [["Price_ROC"], {}],
        "calculate_volume_rate_of_change": [["Volume_ROC"], {}],
        "calculate_volume_ratio": [["Volume_Ratio"], {"period": 20, "ema": True}],
        "calculate_stoch_rsi": [["Stoch_RSI_K", "Stoch_RSI_D"], {"rsi_period": 20, "k_period": 20, "smooth_k": 5, "smooth_k": 5, "ema": True}],
        "calculate_trailing_linear_reg_params": [["Reg_Coef", "RMSE"], {"period": 25, "col_name": "cum_prod"}]
            },
    'strategy_params':{
        "last_features_window": 7,
        "candles_between_pump": 30,
        "validation_window": 20, # candles
        "min_yield": -20, # %
        "max_yield": 20,
        "first_yield": 3
                    }          
    }
    
    
    # Generate features
    print('Start to generate features')
    eval_features_df = add_features(eval_candels_df, exp_config=EXPERIMENT_CONFIG)
    
    
    # Load model
    model = load(os.path.join(config.MODELS_PATH, "xgb_model_new.joblib"))
    
    
    # Predict for today
    print('Start to predict')
    eval_features_df_today = eval_features_df.loc[eval_features_df.index.max()].copy()
    eval_features_df_today.loc[:, 'proba'] = model.best_estimator_.predict_proba(eval_features_df_today.loc[:, model.feature_names_in_])[:, 1]
    eval_features_df_today.loc[:, 'predict'] = model.best_estimator_.predict(eval_features_df_today.loc[:, model.feature_names_in_])
    
    # Find best symbols
    N_BEST_SYMBOLS = 50
    best_symbols = (eval_features_df_today.loc[eval_features_df_today.predict ==1 ,['Symbol', 'predict', 'proba']]
                                         .sort_values('proba', ascending=False)
                                         .head(N_BEST_SYMBOLS)
                   )
    
    # Make plots
    plots_buffer = utils.create_plot_best_symbols(eval_candels_df, best_symbols, show=False)
    
    return plots_buffer, best_symbols.Symbol.values.tolist()