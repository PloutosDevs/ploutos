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


def get_candle_patterns(candles_df: pd.DataFrame, pinbar: float = 0.2, inout: float = 0.2,
                        engulf: float = 0.2) -> pd.DataFrame:
    """
    Return original DataFrame with candles patterns pinbar, engulf, inout

    All params set ratio body candle to full length candle. As lower as more patterns.

    params:
        candles_df - High, Low, Close, Open, Volume values
        pinbar - Coef for pinbar
        inout - Coef for inout
        engulf - Coef for engulf

    return:
        DataFrame
    """

    candles_df = candles_df.copy()

    candles_df.loc[:, ['bull_pin', 'bear_pin', 'in_bar', 'out_bar', 'bull_engulf', 'bear_engulf']] = 0

    for i in range(3, len(candles_df)):
        current = candles_df.iloc[i, :]
        prev = candles_df.iloc[i - 1, :]
        prev_2 = candles_df.iloc[i - 2, :]
        prev_3 = candles_df.iloc[i - 3, :]
        real_body = abs(current['Open'] - current['Close'])
        candle_range = current['High'] - current['Low']
        real_body_prev = abs(prev['Open'] - prev['Close'])
        candle_range_prev = prev['High'] - prev['Low']
        idx = candles_df.index[i]

        # Bullish pinbar
        # Body candle is small, but length is big"""
        candles_df.loc[idx, 'bull_pin'] = int(
            real_body / candle_range >= pinbar and
            current['Close'] > current['Low'] + candle_range / 3 and
            current['Low'] < prev['Low'] and
            prev['Low'] < prev_2['Low']
        )

        # Bearish pinbar
        candles_df.loc[idx, 'bear_pin'] = int(
            real_body / candle_range >= pinbar and
            current['Close'] < current['High'] - candle_range / 3 and
            current['High'] > prev['High'] and
            prev['High'] > prev_2['High']
        )

        # Inside bar
        # Current candle inside range of previous candle
        candles_df.loc[idx, 'in_bar'] = int(
            current['High'] < max(prev['Low'], prev['High']) and
            current['Low'] > min(prev['Low'], prev['High']) and
            real_body_prev / candle_range_prev > inout and
            real_body / candle_range > inout
        )

        # Outside bar
        # Current candle outside range of previous candle
        candles_df.loc[idx, 'out_bar'] = int(
            current['High'] > max(prev['Low'], prev['High']) and
            current['Low'] < min(prev['Low'], prev['High']) and
            real_body_prev / candle_range_prev > inout and
            real_body / candle_range > inout
        )

        # Bullish engulfing
        # Current candle engulfs previous candle
        candles_df.loc[idx, 'bull_engulf'] = int(
            current['High'] > prev['High'] and
            current['Low'] < prev['Low'] and
            real_body >= engulf * candle_range and
            current['Close'] > current['Open'] and
            prev['Close'] < prev['Open']
        )

        # Bearish engulfing
        candles_df.loc[idx, 'bear_engulf'] = int(
            current['High'] > prev['High'] and
            current['Low'] < prev['Low'] and
            real_body >= engulf * candle_range and
            current['Close'] < current['Open'] and
            prev['Close'] > prev['Open']
        )

    return candles_df
