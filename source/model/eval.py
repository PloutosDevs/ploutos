from source.data.get.binance_prices import get_binance_symbols, compose_binance_candles_df
from source.data.process.compose_features import add_features
from joblib import load
import json
import os
import config
from source import utils
import pandas as pd


def eval_model(
        valuation_date: pd.Timestamp = pd.Timestamp.today(tz=config.DEFAULT_TZ).normalize(),
        candles_period: int = 90,
        best_symbols_offset: int = 50,
        model_name: str = config.PROD_MODEL
):
    """

    params:
        valuation_date - Date of signals
        candles_period - Additional candles window for calculation features depended on history
        best_symbols_offset - Best signals
        model_name - Model name

    return:
        DataFrame
    """

    # Get data
    print('Start to get candels')
    BINANCE_SYMBOLS = get_binance_symbols(only_usdt=True)
    # BINANCE_SYMBOLS = ['BTCUSDT', 'LOOMUSDT', 'HIFIUSDT', 'VICUSDT', 'RPLUSDT', 'ATAUSDT']

    candles_start_date = (valuation_date - pd.Timedelta(days=candles_period)).strftime("%Y-%m-%d")
    candles_end_date = valuation_date.strftime("%Y-%m-%d") + " 23:59:59.999999"

    eval_candles_df = compose_binance_candles_df(BINANCE_SYMBOLS, start_time=candles_start_date,
                                                 end_time=candles_end_date, interval='1d')
    print('Got candels successfully')


    print('Start getting model')
    # Load model
    model, experiment_config = load_model_from_path(model_name=model_name)
    print('Model has been loaded')

    # Generate features
    print('Start to generate features')
    eval_features_df = add_features(eval_candles_df, exp_config=experiment_config)
    print('Features were generated')

    # Predict for today
    print('Start to predict')
    eval_features_df_today = eval_features_df.loc[
        (eval_features_df.index == valuation_date) &
        (eval_features_df["yield_before_pump"] >= experiment_config["strategy_params"]["first_yield"])
    ].copy()

    # If there aren't signals
    if eval_features_df_today.empty:
        print('There are not any first_yield symbols')
        return b"", []

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
    
    # If there aren't signals
    if best_symbols.empty:
        print('There are not any 1 class predictions')
        return b"", []
    
    # Make plots
    plots_buffer = utils.create_plot_best_symbols(eval_candles_df, best_symbols, show=False)
    
    print('Prediction ended')
    return plots_buffer, best_symbols.Symbol.values.tolist()


def load_model_from_path(model_name: str):

    model_path = os.path.join(config.PROD_MODEL_PATH, model_name)

    # load config
    with open(os.path.join(model_path, 'exp_config.json')) as json_file:
        enp_config = json.load(json_file)
            
    # Load model
    model = load(os.path.join(model_path, "model.joblib"))
            
    return model, enp_config
