import pandas as pd


def get_finish_abnormal_moves(candles_df: pd.DataFrame, mean_window: int = 14, std_mult: float = 1) -> pd.DataFrame:
    """
    Return original DataFrame with cols after_dump and after_pump

    New values equal 1 or 0 where 1 is first day of positive or negative yield after durable abnormal dump and pump

    params:
        candles_df - High, Low, Close, Open, Volume values
        mean_window - Window to define mean body length. Is taken before last candle - 5.
        std_mult - Border for defining abnormal candle is mean_window - std * std_mult

    return:
        DataFrame
    """

    candles_df = candles_df.copy()

    candles_df["dump"] = 0
    candles_df["after_dump"] = 0
    candles_df["pump"] = 0
    candles_df["after_pump"] = 0

    for i in range(mean_window + 5, len(candles_df)):

        abnormal_dump = 0
        abnormal_pump = 0
        idx = candles_df.index[i]

        sample = candles_df.iloc[i - mean_window + 5:i]

        mean_body = sample["Close"].subtract(sample["Open"]).abs().mean()
        std_body = sample["Close"].subtract(sample["Open"]).abs().std()

        # Define end dumps
        j = i
        while abs(candles_df.iloc[j]["Close"] - candles_df.iloc[j]["Open"]) >= mean_body + std_body * std_mult and \
                candles_df.iloc[j]["Close"] < candles_df.iloc[j]["Open"]:
            abnormal_dump += 1
            j -= 1

        if abnormal_dump >= 3:
            candles_df.loc[idx, 'dump'] = 1

        # Define end pumps
        j = i
        while abs(candles_df.iloc[j]["Close"] - candles_df.iloc[j]["Open"]) >= mean_body + std_body * std_mult and \
                candles_df.iloc[j]["Close"] > candles_df.iloc[j]["Open"]:
            abnormal_pump += 1
            j -= 1

        if abnormal_pump >= 3:
            candles_df.loc[idx, 'pump'] = 1

    dumps = candles_df[candles_df["dump"] == 1]

    for idx in dumps.index:

        pos_yield = candles_df.loc[idx:]["Close"].pct_change()
        pos_yield = pos_yield.loc[pos_yield >= 0]

        if not pos_yield.empty and len(pos_yield) >= 3:

            # Second day with positive yield
            pos_yield_index = pos_yield.index[2]

            candles_df.loc[pos_yield_index, "after_dump"] = 1

    pumps = candles_df[candles_df["pump"] == 1]

    for idx in pumps.index:

        neg_yield = candles_df.loc[idx:]["Close"].pct_change()
        neg_yield = neg_yield.loc[neg_yield < 0]

        if not neg_yield.empty and len(neg_yield) >= 3:

            # Second day with negative yield
            neg_yield_index = neg_yield.index[2]

            candles_df.loc[neg_yield_index, "after_pump"] = 1

    candles_df = candles_df.drop("dump", axis=1)
    candles_df = candles_df.drop("pump", axis=1)

    return candles_df
