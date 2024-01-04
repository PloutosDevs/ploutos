import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


def get_simple_linear_regr_params(data: pd.Series):
    """
    Get coef and rmse
    """

    x, y = np.array(range(len(data.values))), data.values

    x_train = x.reshape(-1, 1)
    y_train = y

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(x_train)

    #     # The coefficients
    #     print("Coefficients: \n", regr.coef_)
    #     # The mean squared error
    #     print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    #     # The coefficient of determination: 1 is perfect prediction
    #     print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

    #     # Plot outputs
    #     plt.scatter(x, y, color="black")
    #     plt.plot(x_test, y_pred, color="blue", linewidth=3)

    #     plt.xticks(())
    #     plt.yticks(())

    #     plt.show()

    return [regr.coef_[0], mean_squared_error(y_train, y_pred) ** 0.5]


def calculate_trailing_linear_reg_params(prices_df, period, col_name):
    """
    Receive DataFrame with specified cols and calculate Trailing linear regression coef and rmse indicators.
    Add values in original DataFrame.

    params:
        prices_df - specified values
        period  - rolling window
        col_name - col name in DataFrame for calculation
    return:
        Add in original DataFrame new cols "Reg_Coef", "RMSE"
    """

    for data in prices_df[col_name].rolling(window=period):
        index = data.index[-1]
        prices_df.loc[index, ["Reg_Coef", "RMSE"]] = get_simple_linear_regr_params(data)

    return
