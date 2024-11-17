import logging
from source.data.get.binance_prices import get_binance_symbols, compose_binance_candles_df
from source.data.process.compose_features import add_features
from joblib import load
import json
import os
import config
from source import utils
import pandas as pd

logger = logging.getLogger(__name__)  # Создаём логгер для текущего модуля


def eval_model(
        valuation_date: pd.Timestamp = pd.Timestamp.today(tz=config.DEFAULT_TZ).normalize(),
        candles_period: int = 90,
        best_symbols_offset: int = 50
):
    """
    params:
        valuation_date - Date of signals
        candles_period - Additional candles window for calculation features depended on history
        best_symbols_offset - Best signals

    return:
        DataFrame
    """

    try:
        # Get data
        logger.info('Starting to get candles')
        BINANCE_SYMBOLS = get_binance_symbols(only_usdt=True)
        # BINANCE_SYMBOLS = ['BTCUSDT', 'LOOMUSDT', 'HIFIUSDT', 'VICUSDT', 'RPLUSDT', 'ATAUSDT']

        candles_start_date = (valuation_date - pd.Timedelta(days=candles_period)).strftime("%Y-%m-%d")
        candles_end_date = valuation_date.strftime("%Y-%m-%d") + " 23:59:59.999999"

        eval_candles_df = compose_binance_candles_df(BINANCE_SYMBOLS, start_time=candles_start_date,
                                                     end_time=candles_end_date, interval='1d')
        logger.info('Candles data retrieved successfully')

        # Load model
        logger.info('Loading model')
        model, experiment_config = load_model_from_path()
        logger.info('Model loaded successfully')

        # Generate features
        logger.info('Generating features')
        eval_features_df = add_features(eval_candles_df, exp_config=experiment_config)
        logger.info('Features generated')

        # Predict for today
        logger.info('Starting prediction')
        eval_features_df_today = eval_features_df.loc[
            (eval_features_df.index == valuation_date) &
            (eval_features_df["yield_before_pump"] >= experiment_config["strategy_params"]["first_yield"])
        ].copy()

        # If there aren't signals
        if eval_features_df_today.empty:
            logger.warning('No symbols meet first_yield criteria')
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
            logger.warning('No class 1 predictions found')
            return b"", []

        # Make plots
        plots_buffer = utils.create_plot_best_symbols(eval_candles_df, best_symbols, show=False)

        logger.info('Prediction completed')
        return plots_buffer, best_symbols.Symbol.values.tolist()

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        raise

def load_model_from_path():

    model_path = config.PROD_MODEL_PATH

    # load config
    logger.info(f'Loading model config from {model_path}')
    with open(os.path.join(model_path, 'exp_config.json')) as json_file:
        enp_config = json.load(json_file)

    # Load model
    logger.info('Loading model object')
    model = load(os.path.join(model_path, "model.joblib"))

    logger.info('Model and config loaded successfully')

    return model, enp_config
