
def get_sup_res_levels(candles_df, dev=0.05):
    """
    Return original DataFrame with supporting and resistance levels.

    params:
        candles_df - High, Low, Close, Open, Volume values
        dev - Set frames in which different levels are treated like equals. Set as % between levels.
    """

    candles_df = candles_df.copy()

    res_list = []
    levels_prices = []
    support_list = []

    for i in range(len(candles_df) - 2):

        if i > 1:

            cond1 = candles_df['High'].iloc[i] > candles_df['High'].iloc[i - 1]
            cond2 = candles_df['High'].iloc[i] > candles_df['High'].iloc[i + 1]
            cond3 = candles_df['High'].iloc[i + 1] > candles_df['High'].iloc[i + 2]
            cond4 = candles_df['High'].iloc[i - 1] > candles_df['High'].iloc[i - 2]

            cond1_ = candles_df['Low'].iloc[i] < candles_df['Low'].iloc[i - 1]
            cond2_ = candles_df['Low'].iloc[i] < candles_df['Low'].iloc[i + 1]
            cond3_ = candles_df['Low'].iloc[i + 1] < candles_df['Low'].iloc[i + 2]
            cond4_ = candles_df['Low'].iloc[i - 1] < candles_df['Low'].iloc[i - 2]

            if (cond1 and cond2 and cond3 and cond4):

                is_exist = False

                for k in levels_prices:
                    if abs(candles_df['High'].iloc[i] / k - 1) <= dev:
                        is_exist = True
                        break
                if not is_exist:
                    res_list.append(candles_df['High'].iloc[i])
                    levels_prices.append(candles_df['High'].iloc[i])

            if (cond1_ and cond2_ and cond3_ and cond4_):

                is_exist = False

                for k in levels_prices:
                    if abs(candles_df['Low'].iloc[i] / k - 1) <= dev:
                        is_exist = True
                        break
                if not is_exist:
                    support_list.append(candles_df['Low'].iloc[i])
                    levels_prices.append(candles_df['Low'].iloc[i])

    return res_list, support_list


def get_breakdown_and_bounce_levels(candles_df, dev=0.05, window=90):

    candles_df = candles_df.copy()

    for roll in candles_df.rolling(window):

        if len(roll) != window:
            continue

        res_list, suppor_list = get_sup_res_levels(roll, dev=dev)

        ind_0 = roll.index[-1]
        ind_1 = roll.index[-2]
        ind_2 = roll.index[-3]

        for sup_level in suppor_list:

            breakdown = (
                roll.loc[ind_0, "Close"] < sup_level and
                roll.loc[ind_1, "Close"] < sup_level and
                roll.loc[ind_0, "Close"] < roll.loc[ind_1, "Close"] and
                roll.loc[ind_2, "Close"] >= sup_level
            )

            bounce = (
                roll.loc[ind_0, "Close"] > sup_level and
                roll.loc[ind_1, "Close"] > sup_level and
                roll.loc[ind_0, "Close"] > roll.loc[ind_1, "Close"] and
                roll.loc[ind_2, "Close"] <= sup_level
            )

            if breakdown:
                candles_df.loc[ind_0, "bd_sup"] = 1

            if bounce:
                candles_df.loc[ind_0, "bn_sup"] = 1

        for res_level in res_list:

            breakdown = (
                roll.loc[ind_0, "Close"] > res_level and
                roll.loc[ind_1, "Close"] > res_level and
                roll.loc[ind_0, "Close"] > roll.loc[ind_1, "Close"] and
                roll.loc[ind_2, "Close"] <= res_level
            )

            bounce = (
                roll.loc[ind_0, "Close"] < res_level and
                roll.loc[ind_1, "Close"] < res_level and
                roll.loc[ind_0, "Close"] < roll.loc[ind_1, "Close"] and
                roll.loc[ind_2, "Close"] >= res_level
            )

            if breakdown:
                candles_df.loc[ind_0, "bd_res"] = 1

            if bounce:
                candles_df.loc[ind_0, "bn_res"] = 1

    candles_df = candles_df.fillna(0)

    return candles_df
