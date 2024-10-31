import pandas as pd
from tqdm import tqdm
import numpy as np
import xgboost as xgb
from joblib import dump
import datetime
from copy import deepcopy
import json
import os

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from source.data.process.compose_features import add_features
from source.utils import optimal_threshold

import config


def generate_data_set(data: pd.DataFrame, strategy_config: dict, drop_bad_values: bool) -> pd.DataFrame:
    """
    Prepares data set for training model from DataFrame with features

    params:
        data: DataFrame with features
        strategy_config:
    return:
        Balanced DataFrame with X and Y
    """

    data_processing = strategy_config["data_processing"]
    strategy_params = strategy_config["strategy_params"]

    data_set = []

    symbols = data["Symbol"].unique()

    for symbol in tqdm(symbols):

        # Get sample of prices by ticker
        sample = data[data["Symbol"] == symbol].copy()

        # Sort by date
        sample = sample.sort_index()

        # Define pumps
        outliers_values = sample[
            (sample["cum_prod"] >= strategy_params["max_yield"]) &
            (sample["yield_before_pump"] >= strategy_params["first_yield"])
        ].copy()

        # If there isn't any pump - skip symbol
        if outliers_values.empty:
            continue

        # Filter pumps by appropriate num of candles between pumps
        last_dt = sample.index[0]
        for dt in outliers_values.index:
            outliers_values.loc[dt, "candles_delta"] = len(sample.loc[last_dt:dt])
            last_dt = dt

        # Get pumps
        outliers_values = outliers_values[outliers_values["candles_delta"] >= strategy_params["candles_between_pump"]]
        outliers_values["data_type"] = (
            f"first_yield>{strategy_params['first_yield']}_cum_prod>{strategy_params['max_yield']}"
        )

        # Get not pumps for diluting data by 0 class in model
        other_values_first = sample[
            (sample["cum_prod"] < strategy_params["max_yield"]) & (sample["cum_prod"] >= 0) &
            (sample["yield_before_pump"] >= strategy_params["first_yield"])
        ].copy()

        other_values_first["data_type"] = (
            f"first_yield>{strategy_params['first_yield']}_cum_prod>0<{strategy_params['max_yield']}"
        )

        other_values_second = sample[
            (sample["cum_prod"] < 0) &
            (sample["yield_before_pump"] >= strategy_params["first_yield"])
        ].copy()

        other_values_second["data_type"] = f"first_yield>{strategy_params['first_yield']}_cum_prod<0"

        # other_values_third = sample[(sample["yield_before_pump"] < strategy_params["first_yield"])].copy()
        # other_values_third["data_type"] = "first_yield<0"

        null_values = pd.concat([other_values_first, other_values_second])  # other_values_third

        # Mark up classes
        outliers_values["class"] = 1
        null_values["class"] = 0

        # Sample 0 (not pump) values
        sample_num = min(int(len(outliers_values) * data_processing["sample_multiplier"]), len(null_values))

        all_values = pd.concat([outliers_values, null_values.sample(sample_num)])

        for date, row in all_values.iterrows():

            data_set_value = (
                sample.iloc[sample.index.get_loc(date) - (strategy_params["validation_window"])].copy()
            )

            data_set_value.loc["class"] = row["class"]
            data_set_value.loc["data_type"] = row["data_type"]

            data_set.append(data_set_value.to_frame().T)

    data_set = pd.concat(data_set).sort_index()

    # Drop useless cols
    data_set = data_set.drop(data_processing["drop_fields"], axis=1)

    if drop_bad_values:
        data_set = data_set.dropna()
        data_set = data_set[np.all(data_set != np.inf, axis=1)]

    return data_set


def prepare_data_sets(candles_df: pd.DataFrame, strategy_config: dict, split_coef: float = 0.8):
    """
    Gets candles, adds features, generates and splits data sets

    params:
        candles_df - DataFrame with High, Low, Close, Open, Volume values
        strategy_config - Config of strategy
        split_coef - Train/Test split coefficient

    return:
        x_train, x_test, y_train, y_test
    """
    # Calculate features
    features_df = add_features(candles_df, strategy_config).sort_index()

    # Define index for split in order to avoid containing equal dates in train and test
    split_index = len(features_df[features_df.index < features_df.iloc[int(len(features_df) * split_coef)].name])

    # Split on train and test
    train_set = generate_data_set(features_df.iloc[:split_index], strategy_config, drop_bad_values=True)
    test_set = generate_data_set(features_df.iloc[split_index:], strategy_config, drop_bad_values=True)

    feature_names = train_set.drop(["Symbol", "class", "data_type"], axis=1).columns

    x_train, x_test, y_train, y_test = (
        train_set.loc[:, feature_names].astype('float'),
        test_set.loc[:, feature_names].astype('float'),
        train_set.loc[:, 'class'].astype('int'),
        test_set.loc[:, 'class'].astype('int'),
    )

    return x_train, x_test, y_train, y_test


def train_model(x_train: pd.DataFrame, y_train: pd.Series, xgb_params: dict):
    """
    Gets data sets and trains model.

    params:
        x_train: DataFrame with features
        y_train: Series with classes
        xgb_params - Params for grid search
        model_name - Model name with .joblib extension

    return:
        Save and return model
    """

    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

    kfold = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

    model = GridSearchCV(
        xgb_model,
        param_grid=xgb_params,
        cv=kfold,
        verbose=1,
        n_jobs=6
    )

    model.fit(x_train, y_train)

    return model


def calculate_model_metrics(model: GridSearchCV, x_test: pd.DataFrame, y_test: pd.Series,
                            proba: bool = False,
                            save_dir: str = None
                            ) -> pd.Series:
    """
    Returns predict and shows up metrics

    params:
        model: Model object
        x_test: Features
        y_test: Classes
        proba: Fit optimal F1 score threshold for proba if true

    return:
        Predict classes
    """

    if not proba:
        predict = model.best_estimator_.predict(x_test)
    else:
        predict_proba = model.best_estimator_.predict_proba(x_test)
        rate, _ = optimal_threshold(predict_proba[:, 1])
        predict = (predict_proba[:, 1] >= rate) * 1

    if save_dir:
        METRICS_FILE_NAME = 'metrics.txt'
        
        with open(os.path.join(save_dir, METRICS_FILE_NAME), 'a') as record_file:
            record_file.write(str(classification_report(y_test, predict) + '\n'))
            
            record_file.write(str(confusion_matrix(y_test, predict)))
            
        
    else:
        print(classification_report(y_test, predict))
        print("accuracy: ", accuracy_score(y_test, predict))
        print(confusion_matrix(y_test, predict))
         # plot_confusion_matrix(y_test, predict, classes=[0, 1])

    return predict



def save_model(model, exp_config, x_test, y_test) -> None:
    # Make new model folder
    new_folder_num = str(len(os.listdir(config.MODELS_PATH)) + 1)
    now = datetime.date.today()

    exp_folder = os.path.join(config.MODELS_PATH, "model_" + new_folder_num + '_' + now.strftime('%Y_%m_%d'))
    os.mkdir(exp_folder)
    print(f"Save new model in {exp_folder}")

    # Save config
    CONFIG_ = deepcopy(exp_config)
    with open(os.path.join(exp_folder, "exp_config.json"), "w") as outfile:
        json.dump(CONFIG_, outfile, skipkeys=True)

    # Save model
    model_name = "model.joblib"
    dump(model, os.path.join(exp_folder, model_name))

    # Save metrics
    _ = calculate_model_metrics(model, x_test, y_test, proba=False, save_dir=exp_folder)
    
    print(f"Model {model_name} saved in {exp_folder}")
    
    
    
