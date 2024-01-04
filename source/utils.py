import pandas as pd
import numpy as np
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config


def get_time_slide_window(interval, exchange, limit=500):
    """
    Max time for adding to previous time for getting candles in MS (s * 1000)

    Limit is quantity of values which can be taken from api
    """

    if exchange == "binance":
        candles_intervals = {
            '1s': 1000 * limit,  # second
            '1m': 60 * 1000 * limit,
            '3m': 3 * 60 * 1000 * limit,
            '5m': 5 * 60 * 1000 * limit,
            '15m': 15 * 60 * 1000 * limit,
            '30m': 30 * 60 * 1000 * limit,
            '1h': 60 * 60 * 1000 * limit,
            '2h': 2 * 60 * 60 * 1000 * limit,
            '4h': 4 * 60 * 60 * 1000 * limit,
            '6h': 6 * 60 * 60 * 1000 * limit,
            '8h': 8 * 60 * 60 * 1000 * limit,
            '12h': 12 * 60 * 60 * 1000 * limit,
            '1d': 24 * 60 * 60 * 1000 * limit,
            '3d': 3 * 24 * 60 * 60 * 1000 * limit,
            '1w': 7 * 24 * 60 * 60 * 1000 * limit,
            '1M': 28 * 24 * 60 * 60 * 1000 * limit,
        }
    else:
        raise ValueError("There is not such exchange")

    return candles_intervals.get(interval)


def normalize_prices(prices: pd.Series):

    yld = [100]
    for i in prices.pct_change().iloc[1:]:
        yld.append(yld[-1] * (i + 1))

    return yld


def min_max_normalization(data: pd.Series):
    """
    Apply min-max normalization

    return normalized pd.Series
    """

    normalized_data = (data - data.min()) / (data.max() - data.min())

    return normalized_data


def drop_highly_corr_features(df, rate=0.95):

    print("before_drop: ", len(df.columns))

    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than rate
    to_drop = [column for column in upper.columns if any(upper[column] > rate)]

    # Drop features
    df.drop(to_drop, axis=1, inplace=True)

    print("after_drop: ", len(df.columns))

    return df


def expect_f1(y_prob, thres):
    
    idxs = np.where(y_prob >= thres)[0]
    tp = y_prob[idxs].sum()
    fp = len(idxs) - tp
    idxs = np.where(y_prob < thres)[0]
    fn = y_prob[idxs].sum()

    return 2*tp / (2*tp + fp + fn)


def optimal_threshold(y_prob):

    y_prob = np.sort(y_prob)[::-1]
    fls = [expect_f1(y_prob, p) for p in y_prob]
    thres = y_prob[np.argmax(fls)]

    return thres, fls

def get_secrets(key_: str) -> str:
    """Function to get secrets to code

    Args:
        key_ (str): key to get secrets

    Returns:
        str: value of secrets
    """
    with open(config.SECRETS_PATH, mode='r') as f:
        SECRETS = json.loads(f.read())
        return SECRETS[key_]
    

if __name__ == '__main__':
    print(get_secrets('TELEGRAM_BOT_TOKEN'))