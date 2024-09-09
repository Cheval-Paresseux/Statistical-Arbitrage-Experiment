import pandas as pd
import numpy as np
import itertools
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from datetime import datetime, timedelta


# This file objective is to gather all functions that are necessary to make a combination of assets respecting certain criterias


# ------------------------------------------------------------------------------------------------------------------------------------------------


def generate_combinations(
    df: pd.DataFrame, num_assets_per_comb: int, max_common_assets: int
):
    """
    This function aims to generate all possible combinations of assets for a given cluster.

    Args :
        df (DataFrame) -> dataframe of a cluster.
        num_assets_per_comb (int) -> the size of the portfolios we want to use (how many assets we want to trade for one strategy application).
        max_common_assets (int) -> limiting the numbers of combination that contains almost the same assets.

    Returns :
        list (Dataframe) -> a list containing all combination possible for this cluster, each combination is a dataframe containing the exp(sum(log return)) history for each asset.
    """
    # Assets List
    assets = df.columns.tolist()

    # Computing all possible combinations
    combinations = list(itertools.combinations(assets, num_assets_per_comb))

    # Set to store assets already selected
    selected_assets = set()

    # List to store the Dataframes filtered
    list_of_dfs = []

    # Keeping only the combinations that are different by max_common number of assets
    for comb in combinations:
        common_assets = set(comb).intersection(selected_assets)

        if len(common_assets) < max_common_assets:
            list_of_dfs.append(df[list(comb)])
            selected_assets.update(comb)

    return list_of_dfs


# ------------------------------------------------------------------------------------------------------------------------------------------------


def reg_lin_multiple(df: pd.DataFrame):
    """
    Performs a multiple linear regression on the DataFrame's columns.

    Returns:
        dict: A dictionary containing the regression coefficients and residuals.
              - 'coeff': Coefficients of the independent variables (the first coefficeint is the intercept)
              - 'residuals': The residuals from the regression (actual - predicted values).
    """
    X = df.iloc[:, 1:]  # Independant variable
    Y = df.iloc[:, 0]  # Dependant variable
    model = LinearRegression()
    model.fit(X, Y)

    # Get inverse of coefficients
    coefficients = -model.coef_

    # Get inverse of the intercept
    intercept = -model.intercept_

    # Add 1 at the start for the dependant variable
    coefficients_total = np.append([1], coefficients)
    coefficients_with_intercept = np.append([intercept], coefficients_total)

    # Predict the values of Y using the fitted model
    predictions = model.predict(X)

    # Calculate the residuals (actual Y - predicted Y)
    residuals = Y - predictions

    # Testing for stationarity
    adf_results = adfuller(residuals)

    # Store the coefficients and residuals in a dictionary
    results = {}
    results["coeff"] = coefficients_with_intercept
    results["residuals"] = residuals
    results["adf"] = adf_results

    return results


# ------------------------------------------------------------------------------------------------------------------------------------------------


def estimate_ou_parameters(data):
    """
    Estimate the parameters of an Ornstein-Uhlenbeck process given a time series.

    Parameters:
        data (array-like): A time series of observations (daily observations).

    Returns:
        dict: Estimated parameters of the OU process (mu, theta, sigma, half_life).
    """
    # Estimate mu as the mean of the series
    mu = np.mean(data)

    # Calculate the differences (discrete version of the derivative)
    delta_data = np.diff(data)

    # Estimate theta using ordinary least squares on AR(1) process
    X = data[:-1] - mu  # X_t - mu
    Y = delta_data  # X_{t+1} - X_t
    reg = sm.OLS(Y, X).fit()
    theta = -reg.params[0]

    if theta > 0:
        # Estimate sigma using the residuals from the AR(1) fit
        residuals = reg.resid
        sigma = np.sqrt(np.std(residuals) * 2 * theta)

        # Compute the half-life of mean reversion
        half_life = np.log(2) / theta
    else:
        theta = 0
        sigma = 0
        half_life = 0

    return {"mu": mu, "theta": theta, "sigma": sigma, "half_life": half_life}


# ------------------------------------------------------------------------------------------------------------------------------------------------


def prepare_data_for_backtest(
    data: pd.DataFrame,
    comb_dict: dict,
    cluster_id: int,
    trading_period_start_date: str,
    timeframe: int,
) -> dict:
    """
    This function aims to provide all the data needed for applying signals.

    Args:
        data (DataFrame) -> Contains all raw prices history of our universe.
        comb_dict (dict) -> Dict of a selected combination.
        cluster_id (int) -> After doing this function we are going out of the cluster loop, so we store the information of which cluster this combination comes from.
        trading_period_start_date (str) -> The start date of the trading period in 'YYYY-MM-DD' format.
        timeframe (int) -> The timeframe to calculate rolling window parameters.

    Returns:
        prepared_data (dict) -> Contains all information needed to process a backtest.
    """
    # Convert trading_period_start_date to datetime
    trading_period_start_date = datetime.strptime(trading_period_start_date, "%Y-%m-%d")

    # 1) Getting the history of the assets in our combination
    assets = comb_dict["assets_name"]
    prices_history_of_assets = data[assets]
    df = prices_history_of_assets.copy()

    # Getting the weights computed during the linear Regression
    coeffs = comb_dict["coeffs"]
    weights = coeffs[1:]
    intercept = coeffs[0]

    # 2) We have computed the residuals on the log prices during the Linear Regression, so here we also take the log prices to recreate the residuals
    log_prices = np.log(prices_history_of_assets)
    # Here we recreate the residuals of the Linear Regression, just on a bigger timeframe
    residuals = log_prices.dot(weights) + intercept
    df["residuals"] = residuals

    # 3) We compute the other parameters
    for idx in residuals.index:
        # Ensure idx is in datetime format if needed (it usually is if it's from a DatetimeIndex)
        if isinstance(idx, pd.Timestamp):
            idx_datetime = idx
        else:
            idx_datetime = pd.to_datetime(idx)

        # We compute the parameters only for the trading period
        if idx_datetime < trading_period_start_date:
            df.loc[
                idx,
                [
                    "z_score",
                    "moving_mean",
                    "moving_sigma",
                    "moving_theta",
                    "moving_halflife",
                    "adf_pvalue",
                ],
            ] = 0
        else:
            # First, we get the residuals on the right timeframe (using the timeframe in days)
            residuals_on_timeframe = residuals[
                (residuals.index < idx_datetime)
                & (residuals.index >= idx_datetime - timedelta(days=timeframe))
            ]
            # Then we estimate our parameters
            ou_params = estimate_ou_parameters(residuals_on_timeframe)
            adf_pvalue = adfuller(residuals_on_timeframe)[1]
            mean = ou_params["mu"]
            sigma = ou_params["sigma"]
            theta = ou_params["theta"]
            halflife = ou_params["half_life"]

            # We can now compute our z-score
            if (
                sigma > 0
            ):  # As we estimate sigma using OU process it can be 0 if theta < 0 (which means residuals are not mean-reverting at all)
                last_resid = residuals_on_timeframe.iloc[-1]
                zscore = (last_resid - mean) / sigma
            else:
                zscore = 0

            # Storing it in the dataframe
            df.loc[idx, "z_score"] = zscore
            df.loc[idx, "adf_pvalue"] = adf_pvalue
            df.loc[idx, "moving_mean"] = mean
            df.loc[idx, "moving_sigma"] = sigma
            df.loc[idx, "moving_theta"] = theta
            df.loc[idx, "moving_halflife"] = halflife

    # Store the data in the test_portfolios dictionary
    prepared_data = {
        "weights": weights,
        "cluster": cluster_id,
        "df": df,
    }
    return prepared_data
