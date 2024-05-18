
def calculate_price_rate_of_change(prices_df):
    """
    Receive DataFrame with prices and calculate price rate of change indicator

    params:
        prices_df - High, Low, Close, Open, Volume values
    return:
        Return original DataFrame with new col "Price_ROC"
    """

    prices_df['Price_ROC'] = prices_df['Close'].pct_change() * 100

    return prices_df


def calculate_volume_rate_of_change(prices_df):
    """
    Receive DataFrame with prices and calculate volume rate of change indicator

    params:
        prices_df - High, Low, Close, Open, Volume values
    return:
        Return original DataFrame with new col "Volume_ROC"
    """

    prices_df['Volume_ROC'] = prices_df['Volume'].pct_change() * 100

    return prices_df
