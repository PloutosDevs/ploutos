
def calculate_price_rate_of_change(prices_df):
    """
    Receive DataFrame with prices and calculate price rate of change indicator. Add values in original DataFrame.

    params:
        prices_df - High, Low, Close, Open, Volume values
    return:
        Add in original DataFrame new col "Price_ROC"
    """

    prices_df['Price_ROC'] = prices_df['Close'].pct_change() * 100

    return


def calculate_volume_rate_of_change(prices_df):
    """
    Receive DataFrame with prices and calculate volume rate of change indicator. Add values in original DataFrame.

    params:
        prices_df - High, Low, Close, Open, Volume values
    return:
        Add in original DataFrame new col "Volume_ROC"
    """

    prices_df['Volume_ROC'] = prices_df['Volume'].pct_change() * 100

    return
