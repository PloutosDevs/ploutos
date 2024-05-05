
def calculate_btc_features(candles_df):
    """
    Create features for other symbols from BTC features
    """

    feature_cols = [
        'SuperTrend', 'MACD_Signal_Line', 'MACD', 'MACD_Bar_Charts', 'RSI', 'OBV_Volume_Ratio', 'Upper_distance',
        'Lower_distance', 'CMF', 'Price_ROC', 'Volume_ROC', 'Volume_Ratio', 'Stoch_RSI_K', 'Stoch_RSI_D', 'Reg_Coef',
        'RMSE'
    ]

    btc_features = candles_df[candles_df["Symbol"] == "BTCUSDT"][feature_cols]
    btc_features.columns = btc_features.columns.map(lambda x: "btc_" + x)

    candles_df = candles_df.merge(btc_features, right_index=True, left_index=True, how="left")

    return candles_df
