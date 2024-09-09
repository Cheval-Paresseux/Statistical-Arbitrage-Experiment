import pandas as pd
import numpy as np
import yfinance as yf

# ------------------------------------------------------------------------------------------------------------------------------------------------


class DataLoaderYFinance:
    """
    A class to manage and sanitize financial time series data.
    """

    def __init__(self, df: pd.DataFrame = None) -> None:
        """
        Initializes the DataLoaderYFinance with an optional DataFrame.

        Args:
            df (pd.DataFrame, optional): The DataFrame to initialize with. Defaults to an empty DataFrame.
        """
        self.df = df or pd.DataFrame()

    @staticmethod
    def get_sp500_tickers() -> list:
        """
        Retrieves the list of S&P 500 tickers from Wikipedia.

        Returns:
            list: A list of S&P 500 tickers.
        """
        sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        sp500_table = pd.read_html(sp500_url)[0]
        sp500_tickers = sp500_table["Symbol"].tolist()
        return sp500_tickers

    def load_ticker_data(
        self, tickers: list, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Loads time series data for multiple tickers over a specified date range.

        Args:
            tickers (list): A list of ticker symbols.
            start_date (str): The start date in 'YYYY-MM-DD' format.
            end_date (str): The end date in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: A DataFrame containing time series data for the specified tickers.
        """
        all_data = {}
        for ticker in tickers:
            try:
                # Download data for each ticker
                data = yf.download(
                    ticker, start=start_date, end=end_date, progress=False
                )
                if not data.empty:
                    # Only keep adjusted close prices
                    all_data[ticker] = data["Adj Close"]
            except Exception as e:
                # Handle any errors that occur during download
                print(f"Error downloading data for {ticker}: {e}")

        # Combine all ticker data into a single DataFrame
        self.df = pd.DataFrame(all_data)
        return self.df


def sanitize_time_series_data(df, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Sanitizes the time series data by filling missing values and ensuring consistent length.

    Args:
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A sanitized DataFrame containing time series data for each ticker.
    """
    # Convert dates to datetime and create a date range
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Reindex the DataFrame to ensure it covers the full date range
    df = df.reindex(date_range)

    # Replace infinities with NaNs and interpolate missing values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.interpolate(method="linear", limit_direction="both")
    df = df.ffill().bfill()

    # Drop columns with insufficient data (e.g., columns with any missing values)
    full_data_threshold = len(df)  # Total number of rows
    valid_columns = df.columns[df.notnull().sum() == full_data_threshold]
    df = df[valid_columns]

    return df


# ------------------------------------------------------------------------------------------------------------------------------------------------


def date_cut(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Returns a DataFrame that contains rows between start_date and end_date.

    Args:
        df (pd.DataFrame): The original DataFrame.
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame containing data between the specified dates.
    """
    # Filter data to keep only rows between start_date and end_date
    data_cut1 = df[df.index < pd.to_datetime(end_date)]
    data_cut2 = data_cut1[data_cut1.index >= pd.to_datetime(start_date)]
    return data_cut2
