import pandas as pd
from tqdm import tqdm
import numpy as np
import xgboost as xgb
from joblib import dump
import datetime
from copy import deepcopy
import json
import os

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, make_scorer, recall_score, precision_score
)

from source.data.process.compose_features import add_features
from source.utils import optimal_threshold, drop_highly_corr_features

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

        # Take cols with high cardinal features
        winzored_cols = data_set.loc[:, data_set.apply(lambda x: len(x.unique())) > len(data_set) / 2].columns

        data_set.loc[:, winzored_cols] = (
            data_set.loc[:, winzored_cols].apply(lambda x: x.clip(x.quantile(0.05), x.quantile(0.95)))
        )

        data_set = drop_highly_corr_features(data_set, rate=0.95)

    data_set = data_set.set_index(["Symbol", "data_type"], append=True)
    data_set.index.names = ['date', 'Symbol', 'data_type']

    return data_set


def prepare_data_sets(candles_df: pd.DataFrame, strategy_config: dict, features_df=pd.DataFrame()):
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
    if features_df.empty:
        features_df = add_features(candles_df, strategy_config).sort_index()
    else:
        feature_mask = "|".join(
            sum(
                list(map(lambda x: x[0], strategy_config['features']["symbol_features"].values())) +
                list(map(lambda x: x[0], strategy_config['features']["dates_features"].values())),
                []
            )
        )
        features_df = features_df.loc[:, ~features_df.columns.str.contains(feature_mask)]

    # Define index for split in order to avoid containing equal dates in train and test
    split_index = len(
        features_df[
            features_df.index <
            features_df.iloc[int(len(features_df) * strategy_config['data_processing']['split_coef'])].name
        ]
    )

    # Split on train and test
    train_set = generate_data_set(features_df.iloc[:split_index], strategy_config, drop_bad_values=True)
    test_set = generate_data_set(features_df.iloc[split_index:], strategy_config, drop_bad_values=True)

    # Drop days with lots signals
    train_set = remove_outliers_days(train_set, quantile_level=0.97)

    x_train, x_test, y_train, y_test = (
        train_set.drop('class', axis=1).astype('float'),
        test_set.drop('class', axis=1).astype('float'),
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

    kfold = KFold(n_splits=5, random_state=None, shuffle=False)

    # Create a custom scorer
    custom_scorer = make_scorer(custom_metric, greater_is_better=True)

    model = GridSearchCV(
        xgb_model,
        scoring=custom_scorer,
        param_grid=xgb_params,
        cv=kfold,
        verbose=4,
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


def custom_metric(y_true, y_pred):

    recall_negative = recall_score(y_true, y_pred, pos_label=0)
    recall_positive = recall_score(y_true, y_pred, pos_label=1)
    # precision_positive = precision_score(y_true, y_pred, pos_label=1)

    return recall_positive# * recall_negative # precision_positive


def remove_outliers_days(train_set, quantile_level=0.97):

    # Group by 'date' and sum up the 'class' values for each day
    grouped_class = train_set.groupby(level='date')['class'].sum()

    # Calculate the quantile threshold
    quantile = int(grouped_class.quantile(quantile_level))

    # Identify dates with outlier values that exceed the quantile threshold
    outliers_days = grouped_class[grouped_class >= quantile].index

    # Create a boolean mask to filter rows with 'class' equal to 1 on outlier days
    bool_mask = (train_set.index.get_level_values('date').isin(outliers_days)) & (train_set['class'] == 1)

    # For outlier days, sample a limited number of rows (up to the quantile threshold) to normalize the dataset
    normal_days = train_set[bool_mask].reset_index().groupby('date').sample(quantile).set_index(train_set.index.names)

    # Concatenate the filtered dataset with the sampled normal days and sort by index
    train_set = pd.concat([train_set[~bool_mask], normal_days]).sort_index()

    return train_set
